#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# ///

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "yuwp-asr"
DEFAULT_METALLIB = REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "mlx.metallib"
BUILD_TIMEOUT_SECONDS = 30 * 60


class SkillError(Exception):
    pass


def eprint(*parts: object) -> None:
    print(*parts, file=sys.stderr)


def tail(text: str, max_lines: int = 40) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SkillError(f"Missing required tool: {name}")


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
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            errors="replace",
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
        description="Transcribe local audio files with the repo-local yuwp-asr CLI.",
    )
    parser.add_argument("audio_file", help="Path to the local audio file")
    parser.add_argument(
        "--format",
        choices=["text", "json", "srt", "vtt"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument("--output", help="Write output to a file instead of stdout")
    parser.add_argument("--model", help="Optional model path or repo id")
    parser.add_argument("--language", help="Optional language hint")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include chunk/alignment debug metadata in JSON output",
    )
    return parser.parse_args()


def resolve_binary() -> Path:
    override = os.environ.get("YUWP_ASR_BIN")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_BIN


def should_auto_build(binary: Path) -> bool:
    if not binary.exists():
        return True
    if binary == DEFAULT_BIN and not DEFAULT_METALLIB.exists():
        return True
    return False


def ensure_release_cli(binary: Path) -> None:
    require_tool("swift")
    require_tool("bash")

    if not should_auto_build(binary):
        return

    eprint("Building yuwp-asr release CLI...")
    run_command(
        ["swift", "build", "-c", "release", "--product", "yuwp-asr"],
        label="swift build",
        cwd=REPO_ROOT,
        timeout=BUILD_TIMEOUT_SECONDS,
    )
    run_command(
        ["bash", "scripts/build_mlx_metallib.sh", "release"],
        label="build_mlx_metallib.sh",
        cwd=REPO_ROOT,
        timeout=BUILD_TIMEOUT_SECONDS,
    )

    if not binary.exists():
        raise SkillError(f"yuwp-asr binary not found after build: {binary}")
    if binary == DEFAULT_BIN and not DEFAULT_METALLIB.exists():
        raise SkillError(f"mlx.metallib not found after build: {DEFAULT_METALLIB}")


def build_transcribe_command(args: argparse.Namespace, binary: Path, audio_file: Path) -> list[str]:
    command = [str(binary), "transcribe", str(audio_file)]

    if args.format != "text":
        command += ["--format", args.format]
    if args.output:
        command += ["--output", args.output]
    if args.model:
        command += ["--model", args.model]
    if args.language:
        command += ["--language", args.language]
    if args.debug:
        command.append("--debug")

    return command


def main() -> int:
    try:
        args = parse_args()
        audio_file = Path(args.audio_file).expanduser().resolve()
        if not audio_file.exists() or not audio_file.is_file():
            raise SkillError(f"Audio file not found: {audio_file}")

        binary = resolve_binary()
        ensure_release_cli(binary)

        result = run_command(
            build_transcribe_command(args, binary, audio_file),
            label="yuwp-asr transcribe",
            cwd=REPO_ROOT,
        )

        if result.stdout:
            sys.stdout.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if result.stderr:
            eprint(result.stderr.rstrip())
        return 0
    except SkillError as exc:
        eprint(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
