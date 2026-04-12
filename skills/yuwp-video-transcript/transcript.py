#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# ///

"""Get a YouTube transcript with yt-dlp captions first, then Yuwp fallback."""

from __future__ import annotations

import argparse
import hashlib
import html
import http.client
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

OUTPUT_DIR = Path("/tmp/yuwp-video-transcripts")
TRANSCRIBE_TIMEOUT_SECONDS = 60 * 60
SUBTITLE_TIMESTAMP_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})[,\.]\d{3}\s+-->\s+")


class SkillError(Exception):
    pass


def eprint(*parts: object) -> None:
    print(*parts, file=sys.stderr)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SkillError(f"Missing required tool: {name}")


def tail(text: str, max_lines: int = 30) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def run_command(command: list[str], *, label: str) -> str:
    result = subprocess.run(command, capture_output=True, text=True, errors="ignore")
    if result.returncode != 0:
        details = tail(result.stderr or result.stdout)
        message = f"{label} failed"
        if details:
            message += f":\n{details}"
        raise SkillError(message)
    return (result.stdout or "") + (result.stderr or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a YouTube transcript with yt-dlp captions first, then Yuwp fallback.",
    )
    parser.add_argument("video_url", help="YouTube video URL")
    parser.add_argument(
        "--hq",
        "--high-quality",
        action="store_true",
        dest="force_hq",
        help="Skip captions and force Yuwp audio transcription",
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Generate SRT subtitles with Yuwp /v1/audio/subtitles",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Override Yuwp server URL (default: $YUWP_SERVER_URL or http://localhost:9748)",
    )
    return parser.parse_args()


def cache_key_for_url(video_url: str) -> str:
    parsed = urlparse(video_url)
    host = (parsed.hostname or "").lower()

    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
        if candidate:
            return candidate

    if host.endswith("youtube.com"):
        query = parse_qs(parsed.query)
        if query.get("v"):
            return query["v"][0]
        parts = [part for part in parsed.path.split("/") if part]
        for marker in ("shorts", "embed", "live"):
            if marker in parts:
                index = parts.index(marker)
                if index + 1 < len(parts):
                    return parts[index + 1]

    digest = hashlib.sha256(video_url.encode("utf-8")).hexdigest()[:16]
    return f"video_{digest}"


def normalize_srt_to_text(content: str) -> str:
    content = content.strip()
    if not content:
        return ""

    blocks = re.split(r"\r?\n\s*\r?\n", content)
    lines_out: list[str] = []
    previous_text: str | None = None

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        timestamp_index: int | None = None
        if len(lines) >= 2 and SUBTITLE_TIMESTAMP_RE.match(lines[1]):
            timestamp_index = 1
        elif SUBTITLE_TIMESTAMP_RE.match(lines[0]):
            timestamp_index = 0

        if timestamp_index is None:
            continue

        match = SUBTITLE_TIMESTAMP_RE.match(lines[timestamp_index])
        if not match:
            continue

        text_lines = lines[timestamp_index + 1 :]
        if not text_lines:
            continue

        text = " ".join(text_lines)
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("\\N", " ")
        text = re.sub(r"\{\\an\d+\}", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text or text == previous_text:
            continue

        lines_out.append(f"[{match.group(1)}] {text}")
        previous_text = text

    return "\n".join(lines_out).strip() + ("\n" if lines_out else "")


def choose_subtitle_file(files: list[Path]) -> Path:
    def score(path: Path) -> tuple[int, int, str]:
        name = path.name
        if name.endswith(".en.srt"):
            rank = 0
        elif ".en-orig." in name:
            rank = 1
        elif ".en." in name:
            rank = 2
        else:
            rank = 3
        return (rank, len(name), name)

    return sorted(files, key=score)[0]


def fetch_subtitles(video_url: str, video_key: str) -> str | None:
    require_tool("yt-dlp")

    with tempfile.TemporaryDirectory(prefix="yuwp-video-subs-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        output_template = tmp_dir / f"{video_key}.%(ext)s"

        command = [
            "yt-dlp",
            "--no-playlist",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-lang",
            "en",
            "--sub-format",
            "srt/vtt/ttml/best",
            "--convert-subs",
            "srt",
            "--no-progress",
            "--remote-components",
            "ejs:github",
            "--extractor-args",
            "youtube:player_client=android",
            "-o",
            str(output_template),
            video_url,
        ]

        try:
            run_command(command, label="Subtitle download")
        except SkillError:
            return None

        subtitle_files = sorted(tmp_dir.glob(f"{video_key}*.srt"))
        if not subtitle_files:
            return None

        subtitle_file = choose_subtitle_file(subtitle_files)
        text = normalize_srt_to_text(subtitle_file.read_text(errors="ignore"))
        return text or None


def ensure_audio_downloaded(video_url: str, audio_path: Path) -> Path:
    require_tool("yt-dlp")
    require_tool("ffmpeg")

    if audio_path.exists() and audio_path.stat().st_size > 0:
        return audio_path

    eprint("Downloading audio...")
    output_template = audio_path.with_suffix(".%(ext)s")

    command = [
        "yt-dlp",
        "-f",
        "bestaudio[ext=m4a]/bestaudio/best",
        "-x",
        "--audio-format",
        "m4a",
        "--audio-quality",
        "0",
        "-o",
        str(output_template),
        "--no-playlist",
        "--no-progress",
        "--remote-components",
        "ejs:github",
        "--extractor-args",
        "youtube:player_client=web",
        video_url,
    ]

    run_command(command, label="Audio download")

    if audio_path.exists() and audio_path.stat().st_size > 0:
        return audio_path

    candidates = sorted(audio_path.parent.glob(f"{audio_path.stem}.*"))
    for candidate in candidates:
        if candidate == audio_path or candidate.suffix == ".part":
            continue
        if candidate.suffix == ".m4a":
            candidate.replace(audio_path)
            return audio_path

        run_command(
            [
                "ffmpeg",
                "-i",
                str(candidate),
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                str(audio_path),
                "-y",
                "-loglevel",
                "error",
            ],
            label="Audio conversion",
        )
        candidate.unlink(missing_ok=True)
        if audio_path.exists() and audio_path.stat().st_size > 0:
            return audio_path

    raise SkillError("Audio download did not produce a usable file")


def open_connection(server_url: str, timeout: int) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlparse(server_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"}:
        raise SkillError(f"Unsupported server URL: {server_url}")
    if not parsed.hostname:
        raise SkillError(f"Invalid server URL: {server_url}")

    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname,
            parsed.port,
            timeout=timeout,
        )
    else:
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)

    base_path = parsed.path.rstrip("/")
    return connection, base_path


def request_text(server_url: str, method: str, path: str, *, timeout: int = 30) -> tuple[int, str]:
    connection, base_path = open_connection(server_url, timeout)
    request_path = f"{base_path}{path}" if base_path else path
    if not request_path.startswith("/"):
        request_path = "/" + request_path

    try:
        connection.request(method, request_path)
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        return response.status, body
    except OSError as exc:
        raise SkillError(f"Server request failed: {exc}") from exc
    finally:
        connection.close()


def ensure_server_ready(server_url: str) -> None:
    status, body = request_text(server_url, "GET", "/v1/info", timeout=10)
    if status >= 400:
        raise SkillError(f"Yuwp server returned {status} for /v1/info: {body.strip()}")


def multipart_post_file(
    server_url: str,
    endpoint: str,
    *,
    file_path: Path,
    fields: dict[str, str],
) -> str:
    connection, base_path = open_connection(server_url, TRANSCRIBE_TIMEOUT_SECONDS)
    request_path = f"{base_path}{endpoint}" if base_path else endpoint
    if not request_path.startswith("/"):
        request_path = "/" + request_path

    boundary = f"----yuwp-{uuid.uuid4().hex}"
    field_chunks: list[bytes] = []
    for name, value in fields.items():
        field_chunks.append(
            (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n"
            ).encode("utf-8")
        )

    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_header = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{file_path.name}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    file_footer = f"\r\n--{boundary}--\r\n".encode("utf-8")

    content_length = sum(len(chunk) for chunk in field_chunks)
    content_length += len(file_header) + file_path.stat().st_size + len(file_footer)

    try:
        connection.putrequest("POST", request_path)
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        connection.endheaders()

        for chunk in field_chunks:
            connection.send(chunk)
        connection.send(file_header)
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        connection.send(file_footer)

        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            raise SkillError(f"Request failed ({response.status}):\n{body.strip()}")
        return body
    except OSError as exc:
        raise SkillError(f"Upload failed: {exc}") from exc
    finally:
        connection.close()


def transcribe_text(server_url: str, audio_path: Path) -> str:
    ensure_server_ready(server_url)
    eprint("Transcribing with Yuwp...")
    return multipart_post_file(
        server_url,
        "/v1/audio/transcriptions",
        file_path=audio_path,
        fields={
            "model": "qwen3-asr",
            "response_format": "text",
        },
    )


def transcribe_srt(server_url: str, audio_path: Path) -> str:
    ensure_server_ready(server_url)
    eprint("Generating SRT with Yuwp...")
    return multipart_post_file(
        server_url,
        "/v1/audio/subtitles",
        file_path=audio_path,
        fields={
            "response_format": "srt",
        },
    )


def main() -> int:
    args = parse_args()
    server_url = args.server or os.environ.get("YUWP_SERVER_URL", "http://localhost:9748")

    video_key = cache_key_for_url(args.video_url)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transcript_path = OUTPUT_DIR / f"{video_key}_transcript.txt"
    srt_path = OUTPUT_DIR / f"{video_key}_transcript.srt"
    audio_path = OUTPUT_DIR / f"{video_key}_audio.m4a"

    if args.srt:
        if srt_path.exists() and srt_path.stat().st_size > 0:
            sys.stdout.write(srt_path.read_text(errors="ignore"))
            return 0

        srt_text = transcribe_srt(server_url, ensure_audio_downloaded(args.video_url, audio_path))
        srt_path.write_text(srt_text)
        sys.stdout.write(srt_text)
        return 0

    if not args.force_hq and transcript_path.exists() and transcript_path.stat().st_size > 0:
        sys.stdout.write(transcript_path.read_text(errors="ignore"))
        return 0

    if not args.force_hq:
        subtitles_text = fetch_subtitles(args.video_url, video_key)
        if subtitles_text:
            transcript_path.write_text(subtitles_text)
            sys.stdout.write(subtitles_text)
            return 0

        eprint("Subtitles unavailable, using Yuwp audio transcription...")

    transcript_text = transcribe_text(server_url, ensure_audio_downloaded(args.video_url, audio_path))
    transcript_path.write_text(transcript_text)
    sys.stdout.write(transcript_text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SkillError as exc:
        eprint(f"Error: {exc}")
        raise SystemExit(1)
