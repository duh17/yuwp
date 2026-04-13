#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# ///

"""Get a YouTube transcript with yt-dlp captions first, then Yuwp ASR fallback."""

from __future__ import annotations

import argparse
import hashlib
import html
import http.client
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path("/tmp/yuwp-video-transcripts")
DEFAULT_SERVER_URL = "http://127.0.0.1:9748"
TRANSCRIBE_TIMEOUT_SECONDS = 60 * 60
BUILD_TIMEOUT_SECONDS = 30 * 60
SERVER_READY_TIMEOUT_SECONDS = 120
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


def run_command(
    command: list[str],
    *,
    label: str,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="ignore",
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SkillError(f"{label} timed out after {timeout}s") from exc

    if result.returncode != 0:
        details = tail(result.stderr or result.stdout)
        message = f"{label} failed"
        if details:
            message += f":\n{details}"
        raise SkillError(message)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a YouTube transcript with yt-dlp captions first, then Yuwp ASR fallback.",
    )
    parser.add_argument("video_url", help="YouTube video URL")
    parser.add_argument(
        "--hq",
        "--high-quality",
        action="store_true",
        dest="force_hq",
        help="Skip captions and force local Yuwp audio transcription",
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Generate SRT subtitles through Yuwp's batch endpoint (auto-starts `yuwp-asr serve` if needed)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model path or repo id passed through to `yuwp-asr serve --model` when starting a local server",
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


def cache_stem(video_url: str, model_spec: str | None) -> str:
    base = cache_key_for_url(video_url)
    if not model_spec:
        return base
    digest = hashlib.sha256(model_spec.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"


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


def built_cli_candidates() -> list[Path]:
    return [
        REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "yuwp-asr",
        REPO_ROOT / ".build" / "release" / "yuwp-asr",
        REPO_ROOT / ".build" / "arm64-apple-macosx" / "debug" / "yuwp-asr",
        REPO_ROOT / ".build" / "debug" / "yuwp-asr",
    ]


def build_yuwp_asr() -> None:
    require_tool("swift")

    eprint("Building yuwp-asr...")
    run_command(
        ["swift", "build", "-c", "release", "--product", "yuwp-asr"],
        label="swift build yuwp-asr",
        cwd=REPO_ROOT,
        timeout=BUILD_TIMEOUT_SECONDS,
    )

    release_metallib = REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "mlx.metallib"
    if release_metallib.exists():
        return

    require_tool("bash")
    run_command(
        ["bash", "scripts/build_mlx_metallib.sh", "release"],
        label="build mlx.metallib",
        cwd=REPO_ROOT,
        timeout=BUILD_TIMEOUT_SECONDS,
    )


def candidate_has_runtime_assets(cli_path: Path) -> bool:
    return cli_path.with_name("mlx.metallib").exists()


def ensure_mlx_metallib(cli_path: Path) -> None:
    metallib_path = cli_path.with_name("mlx.metallib")
    if metallib_path.exists():
        return

    if cli_path.is_relative_to(REPO_ROOT / ".build"):
        configuration = cli_path.parent.name
        eprint(f"Missing {metallib_path.name}; building {configuration} Metal library...")
        run_command(
            ["bash", "scripts/build_mlx_metallib.sh", configuration],
            label=f"build mlx.metallib ({configuration})",
            cwd=REPO_ROOT,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
        if metallib_path.exists():
            return

    raise SkillError(f"Missing required runtime asset: {metallib_path}")


def ensure_yuwp_asr_binary() -> Path:
    override = os.environ.get("YUWP_ASR_BIN")
    if override:
        cli_path = Path(override).expanduser().resolve()
        if not cli_path.exists():
            raise SkillError(f"YUWP_ASR_BIN points to a missing file: {cli_path}")
        ensure_mlx_metallib(cli_path)
        return cli_path

    existing_candidates = [candidate for candidate in built_cli_candidates() if candidate.exists() and os.access(candidate, os.X_OK)]
    for candidate in existing_candidates:
        if candidate_has_runtime_assets(candidate):
            return candidate

    build_yuwp_asr()

    existing_candidates = [candidate for candidate in built_cli_candidates() if candidate.exists() and os.access(candidate, os.X_OK)]
    for candidate in existing_candidates:
        if candidate_has_runtime_assets(candidate):
            return candidate

    for candidate in existing_candidates:
        ensure_mlx_metallib(candidate)
        return candidate

    expected = "\n".join(f"- {candidate}" for candidate in built_cli_candidates())
    raise SkillError(f"Could not find yuwp-asr after build. Checked:\n{expected}")


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


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_server_ready(
    server_url: str,
    *,
    process: subprocess.Popen[str],
    log_path: Path,
    timeout: int = SERVER_READY_TIMEOUT_SECONDS,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            log_tail = tail(log_path.read_text(errors="ignore")) if log_path.exists() else ""
            message = f"yuwp-asr serve exited before becoming ready"
            if log_tail:
                message += f":\n{log_tail}"
            raise SkillError(message)

        try:
            ensure_server_ready(server_url)
            return
        except SkillError:
            time.sleep(0.5)

    log_tail = tail(log_path.read_text(errors="ignore")) if log_path.exists() else ""
    message = f"Timed out waiting for yuwp-asr serve at {server_url}"
    if log_tail:
        message += f":\n{log_tail}"
    raise SkillError(message)


def maybe_reuse_server(model_spec: str | None) -> str | None:
    override = os.environ.get("YUWP_SERVER_URL")
    if override:
        ensure_server_ready(override)
        return override

    if model_spec is not None:
        return None

    try:
        ensure_server_ready(DEFAULT_SERVER_URL)
        return DEFAULT_SERVER_URL
    except SkillError:
        return None


def transcribe_with_yuwp(audio_path: Path, *, output_format: str, model_spec: str | None) -> str:
    server_url = maybe_reuse_server(model_spec)
    if server_url:
        eprint(f"Uploading audio to {server_url}...")
        return multipart_post_file(
            server_url,
            "/v1/audio/transcriptions",
            file_path=audio_path,
            fields={"response_format": output_format},
        )

    cli_path = ensure_yuwp_asr_binary()
    port = pick_free_port()
    server_url = f"http://127.0.0.1:{port}"
    log_path = OUTPUT_DIR / f"yuwp-asr-serve-{port}.log"
    command = [str(cli_path), "serve", "--host", "127.0.0.1", "--port", str(port)]
    if model_spec:
        command += ["--model", model_spec]

    eprint(f"Starting {cli_path.name} serve on {server_url}...")
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=log_handle,
            text=True,
        )
        try:
            wait_for_server_ready(server_url, process=process, log_path=log_path)
            eprint(f"Uploading audio to {server_url}...")
            return multipart_post_file(
                server_url,
                "/v1/audio/transcriptions",
                file_path=audio_path,
                fields={"response_format": output_format},
            )
        finally:
            stop_process(process)


def main() -> int:
    args = parse_args()

    cache_key = cache_stem(args.video_url, args.model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transcript_path = OUTPUT_DIR / f"{cache_key}_transcript.txt"
    srt_path = OUTPUT_DIR / f"{cache_key}_transcript.srt"
    audio_path = OUTPUT_DIR / f"{cache_key}_audio.m4a"

    if args.srt:
        if srt_path.exists() and srt_path.stat().st_size > 0:
            sys.stdout.write(srt_path.read_text(errors="ignore"))
            return 0

        srt_text = transcribe_with_yuwp(
            ensure_audio_downloaded(args.video_url, audio_path),
            output_format="srt",
            model_spec=args.model,
        )
        srt_path.write_text(srt_text)
        sys.stdout.write(srt_text)
        return 0

    if not args.force_hq and transcript_path.exists() and transcript_path.stat().st_size > 0:
        sys.stdout.write(transcript_path.read_text(errors="ignore"))
        return 0

    if not args.force_hq:
        subtitles_text = fetch_subtitles(args.video_url, cache_key)
        if subtitles_text:
            transcript_path.write_text(subtitles_text)
            sys.stdout.write(subtitles_text)
            return 0

        eprint("Subtitles unavailable, using yuwp-asr audio transcription...")

    transcript_text = transcribe_with_yuwp(
        ensure_audio_downloaded(args.video_url, audio_path),
        output_format="text",
        model_spec=args.model,
    )
    transcript_path.write_text(transcript_text)
    sys.stdout.write(transcript_text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SkillError as exc:
        eprint(f"Error: {exc}")
        raise SystemExit(1)
