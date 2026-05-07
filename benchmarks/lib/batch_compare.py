#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "mlx-audio>=0.2.2",
# ]
# ///
"""Composable batch transcription benchmark for Yuwp and reference tools.

This script covers the benchmark use cases we have been using manually:
- compare Yuwp server vs mlx-audio vs qwen_asr on arbitrary audio files
- compare Yuwp VAD chunking vs low-energy fallback on the same long file
- compare different model directories without hard-coded sample sets
- compare the standalone Yuwp CLI (`yuwp-asr transcribe`) against server or other tools

Interface note:
- use `yuwp-asr serve|transcribe`

Examples:
  # Yuwp server vs mlx-audio on one long file, with transcript diffs
  uv run benchmarks/cli.py asr compare \
    --audio /tmp/yuwp-subtitle-bench/video-32k.m4a \
    --tool yuwp --tool mlx-audio \
    --compare-text

  # VAD vs low-energy fallback on the same Yuwp model
  uv run benchmarks/cli.py asr compare \
    --audio /tmp/yuwp-subtitle-bench/video-32k.m4a \
    --tool yuwp \
    --yuwp-chunking vad \
    --yuwp-chunking energy \
    --compare-text

  # Reproduce the practical comparison, but explicitly
  uv run benchmarks/cli.py asr compare \
    --audio /path/to/qwen-asr/samples/jfk.wav \
    --audio /tmp/yuwp-subtitle-bench/video-4m.m4a \
    --audio /tmp/yuwp-subtitle-bench/video-32k.m4a \
    --tool yuwp --tool mlx-audio --tool qwen-asr \
    --qwen-args '-S 30 -W 3' \
    --compare-text \
    --json /tmp/asr-benchmark.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_FIXTURES_DIR = REPO_ROOT / "Tests" / "fixtures"
QWEN_REPO = Path.home() / "workspace" / "qwen-asr"
YUWP_CANONICAL_CLI_BIN = REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "yuwp-asr"
YUWP_METALLIB = REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "mlx.metallib"
DEFAULT_YUWP_MODEL = Path.home() / ".cache" / "huggingface" / "hub" / "models--mlx-community--Qwen3-ASR-0.6B-bf16" / "snapshots" / "eae2b51f96265328f1e7beced788adb0e4536f92"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-ASR-0.6B-bf16"
DEFAULT_QWEN_MODEL = QWEN_REPO / "qwen3-asr-0.6b"
QWEN_BIN = QWEN_REPO / "qwen_asr"


@dataclass(frozen=True)
class Variant:
    id: str
    tool: str
    tool_label: str
    params: dict[str, Any]


@dataclass
class Measurement:
    variant_id: str
    tool: str
    audio: str
    repeat: int
    duration_s: float
    wall_s: float
    rtx: float
    chars: int
    text: str
    startup_s: float | None = None


@dataclass
class Aggregate:
    variant_id: str
    tool: str
    audio: str
    repeats: int
    duration_s: float
    startup_s: float | None
    mean_wall_s: float
    min_wall_s: float
    max_wall_s: float
    weighted_rtx: float
    chars: int
    text: str


@dataclass
class VariantSummary:
    variant_id: str
    tool: str
    cases: int
    repeats: int
    total_audio_s: float
    total_wall_s: float
    weighted_rtx: float
    startup_s: float | None


@dataclass
class TextCompare:
    audio: str
    lhs: str
    rhs: str
    raw_exact: bool
    normalized_exact: bool
    cer: float
    wer: float
    lhs_chars: int
    rhs_chars: int


@dataclass
class PreparedYuwpServer:
    process: subprocess.Popen[str]
    port: int
    startup_s: float
    log_path: str


def iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def audio_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-i", str(path), "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[:500]}")
    return float(result.stdout.strip())


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\.,!?;:\"'`”“’‘()\[\]{}]", "", text)
    return text.strip()


def levenshtein(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    if seq_a == seq_b:
        return 0
    if len(seq_a) == 0:
        return len(seq_b)
    if len(seq_b) == 0:
        return len(seq_a)
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    previous = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a, start=1):
        current = [i]
        for j, item_b in enumerate(seq_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if item_a == item_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def char_error_rate(expected: str, actual: str) -> float:
    if not expected and not actual:
        return 0.0
    return levenshtein(expected, actual) / max(len(expected), 1)


def word_error_rate(expected: str, actual: str) -> float:
    expected_words = expected.split()
    actual_words = actual.split()
    if not expected_words and not actual_words:
        return 0.0
    return levenshtein(expected_words, actual_words) / max(len(expected_words), 1)


def find_free_port() -> int:
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def model_label(path: Path | str) -> str:
    if isinstance(path, str):
        return path.split("/")[-1]
    for part in path.parts:
        if part.startswith("models--"):
            pieces = part.split("--")
            if pieces:
                return pieces[-1]
    return path.name or path.parent.name


def slugify(text: str) -> str:
    text = text.strip()
    if not text:
        return "default"
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._=-]+", "-", text)
    return text.strip("-")[:40] or "default"


def parse_extra_args(raw: str) -> list[str]:
    return shlex.split(raw)


def multipart_request(url: str, file_path: Path) -> bytes:
    boundary = f"----Benchmark{int(time.time() * 1000)}"
    file_bytes = file_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        f"qwen3-asr\r\n"
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        f"json\r\n"
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{file_path.name}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    request = Request(url, data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Content-Length", str(len(body)))
    with urlopen(request, timeout=7200) as response:
        return response.read()


def convert_to_qwen_wav(audio_path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="qwen-benchmark-", suffix=".wav", delete=False)
    tmp.close()
    out = Path(tmp.name)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {audio_path}: {result.stderr.strip()[:500]}")
    return out


def parse_qwen_stderr(stderr: str) -> tuple[float, float]:
    infer_match = re.search(r"Inference:\s*([0-9.]+)\s*ms", stderr)
    rtx_match = re.search(r"\(([0-9.]+)x realtime\)", stderr)
    if not infer_match or not rtx_match:
        raise RuntimeError(f"could not parse qwen_asr timings:\n{stderr[-1000:]}")
    return float(infer_match.group(1)) / 1000.0, float(rtx_match.group(1))


def resolve_yuwp_server_bin() -> Path:
    return YUWP_CANONICAL_CLI_BIN


def start_yuwp_server(model_dir: Path, disable_vad: bool) -> PreparedYuwpServer:
    log_file = tempfile.NamedTemporaryFile(prefix="yuwp-benchmark-", suffix=".log", delete=False)
    log_file.close()
    port = find_free_port()
    server_bin = resolve_yuwp_server_bin()
    command = [
        str(server_bin),
        "serve",
        "--model",
        str(model_dir),
        "--transport",
        "http",
        "--port",
        str(port),
    ]
    if disable_vad:
        command.append("--disable-vad")

    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=open(log_file.name, "w"), stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}/v1/info"
    for _ in range(240):
        try:
            with urlopen(url, timeout=2) as response:
                payload = json.loads(response.read().decode())
                if payload.get("status") == "ready":
                    return PreparedYuwpServer(process=process, port=port, startup_s=time.perf_counter() - started, log_path=log_file.name)
        except Exception:
            pass
        time.sleep(1)

    process.kill()
    process.wait(timeout=5)
    raise RuntimeError(f"Yuwp server failed to become ready. Log: {log_file.name}")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_yuwp_server(audio_path: Path, server: PreparedYuwpServer) -> tuple[str, float, float]:
    url = f"http://127.0.0.1:{server.port}/v1/audio/transcriptions"
    started = time.perf_counter()
    body = multipart_request(url, audio_path)
    wall_s = time.perf_counter() - started
    payload = json.loads(body.decode())
    text = payload.get("text", "")
    duration_s = audio_duration_seconds(audio_path)
    return text, wall_s, duration_s / max(wall_s, 1e-9)


def resolve_yuwp_cli_bin() -> Path:
    return YUWP_CANONICAL_CLI_BIN


def run_yuwp_cli(audio_path: Path, model_dir: Path) -> tuple[str, float, float]:
    cli_bin = resolve_yuwp_cli_bin()
    command = [str(cli_bin), "transcribe", str(audio_path), "--model", str(model_dir), "--format", "json"]
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
    wall_s = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"{cli_bin.name} failed for {audio_path.name}: {process.stderr.strip()[:500]}")
    payload = json.loads(process.stdout)
    return payload["text"], wall_s, float(payload["speedMultiplier"])


def load_mlx_audio_wrapper(model_name: str):
    from mlx_audio.stt import load_model
    started = time.perf_counter()
    wrapper = load_model(model_name)
    return wrapper, time.perf_counter() - started


def run_mlx_audio(audio_path: Path, wrapper) -> tuple[str, float, float]:
    from mlx_audio.stt.generate import generate_transcription
    previous_cwd = Path.cwd()
    started = time.perf_counter()
    os.chdir(tempfile.gettempdir())
    try:
        result = generate_transcription(wrapper, str(audio_path))
    finally:
        os.chdir(previous_cwd)
    wall_s = time.perf_counter() - started
    duration_s = audio_duration_seconds(audio_path)
    text = result.text.strip()
    return text, wall_s, duration_s / max(wall_s, 1e-9)


def run_qwen(audio_path: Path, model_dir: Path, extra_args: list[str]) -> tuple[str, float, float]:
    qwen_audio = convert_to_qwen_wav(audio_path)
    try:
        command = [str(QWEN_BIN), "-d", str(model_dir), "-i", str(qwen_audio), *extra_args]
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, cwd=QWEN_REPO)
        if process.returncode != 0:
            raise RuntimeError(f"qwen_asr failed for {audio_path.name}: {process.stderr.strip()[:500]}")
        infer_s, rtx = parse_qwen_stderr(process.stderr)
        del infer_s
        return process.stdout.strip(), time.perf_counter() - started, rtx
    finally:
        qwen_audio.unlink(missing_ok=True)


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants: list[Variant] = []
    tools = args.tool or []

    yuwp_models = args.yuwp_model or [DEFAULT_YUWP_MODEL]
    yuwp_chunkings = args.yuwp_chunking or ["vad"]
    mlx_models = args.mlx_model or [DEFAULT_MLX_MODEL]
    qwen_models = args.qwen_model or [DEFAULT_QWEN_MODEL]
    qwen_args_sets = args.qwen_args or [""]

    if "yuwp" in tools:
        multiple_models = len(yuwp_models) > 1
        multiple_chunkings = len(yuwp_chunkings) > 1
        for model_dir, chunking in itertools.product(yuwp_models, yuwp_chunkings):
            label = ["yuwp"]
            if multiple_models:
                label.append(model_label(model_dir))
            if multiple_chunkings or chunking != "vad":
                label.append(f"chunking={chunking}")
            variants.append(Variant(
                id="[".join([label[0], ", ".join(label[1:])]) + "]" if len(label) > 1 else "yuwp",
                tool="yuwp",
                tool_label="Yuwp server",
                params={"model_dir": model_dir, "chunking": chunking},
            ))

    if "yuwp-cli" in tools:
        multiple_models = len(yuwp_models) > 1
        for model_dir in yuwp_models:
            label = "yuwp-cli"
            if multiple_models:
                label = f"yuwp-cli[{model_label(model_dir)}]"
            variants.append(Variant(
                id=label,
                tool="yuwp-cli",
                tool_label="yuwp-asr transcribe",
                params={"model_dir": model_dir},
            ))

    if "mlx-audio" in tools:
        multiple_models = len(mlx_models) > 1
        for model_name in mlx_models:
            label = "mlx-audio"
            if multiple_models:
                label = f"mlx-audio[{model_label(model_name)}]"
            variants.append(Variant(
                id=label,
                tool="mlx-audio",
                tool_label="mlx-audio",
                params={"model_name": model_name},
            ))

    if "qwen-asr" in tools:
        multiple_models = len(qwen_models) > 1
        multiple_args = len(qwen_args_sets) > 1 or qwen_args_sets != [""]
        for model_dir, raw_args in itertools.product(qwen_models, qwen_args_sets):
            label = ["qwen-asr"]
            if multiple_models:
                label.append(model_label(model_dir))
            if multiple_args:
                label.append(f"args={raw_args or 'default'}")
            variants.append(Variant(
                id="[".join([label[0], ", ".join(label[1:])]) + "]" if len(label) > 1 else "qwen-asr",
                tool="qwen-asr",
                tool_label="qwen_asr",
                params={"model_dir": model_dir, "raw_args": raw_args, "extra_args": parse_extra_args(raw_args)},
            ))

    return variants


def validate_args(args: argparse.Namespace) -> tuple[list[Path], list[Variant]]:
    if not args.audio:
        raise SystemExit("--audio is required at least once")
    if not args.tool:
        raise SystemExit("--tool is required at least once")

    audio_files = [require_path(path.expanduser().resolve(), "audio file") for path in args.audio]
    variants = build_variants(args)
    if not variants:
        raise SystemExit("no benchmark variants were constructed from the provided flags")

    if any(v.tool == "yuwp" for v in variants):
        require_path(resolve_yuwp_server_bin(), "Yuwp server binary")
        require_path(YUWP_METALLIB, "mlx.metallib")
        for variant in variants:
            if variant.tool == "yuwp":
                require_path(Path(variant.params["model_dir"]), "Yuwp model")

    if any(v.tool == "yuwp-cli" for v in variants):
        require_path(resolve_yuwp_cli_bin(), "yuwp-asr binary")
        require_path(YUWP_METALLIB, "mlx.metallib")
        for variant in variants:
            if variant.tool == "yuwp-cli":
                require_path(Path(variant.params["model_dir"]), "Yuwp model")

    if any(v.tool == "qwen-asr" for v in variants):
        require_path(QWEN_BIN, "qwen_asr binary")
        for variant in variants:
            if variant.tool == "qwen-asr":
                require_path(Path(variant.params["model_dir"]), "qwen_asr model")

    return audio_files, variants


def describe_plan(audio_files: list[Path], variants: list[Variant], repeats: int) -> str:
    lines = ["Plan:", f"- audio files: {len(audio_files)}", f"- variants: {len(variants)}", f"- repeats: {repeats}", f"- total runs: {len(audio_files) * len(variants) * repeats}", "", "Variants:"]
    for variant in variants:
        lines.append(f"- {variant.id}: {variant.tool_label}")
        for key, value in variant.params.items():
            lines.append(f"    {key}: {value}")
    lines.append("")
    lines.append("Audio:")
    for audio in audio_files:
        lines.append(f"- {audio}")
    return "\n".join(lines)


def save_text(save_dir: Path | None, variant_id: str, audio_path: Path, repeat: int, text: str) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{slugify(variant_id)}--{slugify(audio_path.stem)}--r{repeat}.txt"
    (save_dir / filename).write_text(text)


def run_variant(variant: Variant, audio_files: list[Path], repeats: int, save_text_dir: Path | None) -> tuple[list[Measurement], dict[str, Any]]:
    measurements: list[Measurement] = []
    metadata: dict[str, Any] = {}

    if variant.tool == "yuwp":
        server = start_yuwp_server(Path(variant.params["model_dir"]), disable_vad=(variant.params["chunking"] == "energy"))
        metadata["startup_s"] = server.startup_s
        metadata["log_path"] = server.log_path
        try:
            for repeat in range(1, repeats + 1):
                for audio_path in audio_files:
                    eprint(f"[benchmark] {variant.id} :: {audio_path.name} :: repeat {repeat}/{repeats}")
                    text, wall_s, rtx = run_yuwp_server(audio_path, server)
                    duration_s = audio_duration_seconds(audio_path)
                    save_text(save_text_dir, variant.id, audio_path, repeat, text)
                    measurements.append(Measurement(
                        variant_id=variant.id,
                        tool=variant.tool_label,
                        audio=str(audio_path),
                        repeat=repeat,
                        duration_s=duration_s,
                        wall_s=wall_s,
                        rtx=rtx,
                        chars=len(text),
                        text=text,
                        startup_s=server.startup_s,
                    ))
        finally:
            stop_process(server.process)
        return measurements, metadata

    if variant.tool == "yuwp-cli":
        for repeat in range(1, repeats + 1):
            for audio_path in audio_files:
                eprint(f"[benchmark] {variant.id} :: {audio_path.name} :: repeat {repeat}/{repeats}")
                text, wall_s, rtx = run_yuwp_cli(audio_path, Path(variant.params["model_dir"]))
                duration_s = audio_duration_seconds(audio_path)
                save_text(save_text_dir, variant.id, audio_path, repeat, text)
                measurements.append(Measurement(
                    variant_id=variant.id,
                    tool=variant.tool_label,
                    audio=str(audio_path),
                    repeat=repeat,
                    duration_s=duration_s,
                    wall_s=wall_s,
                    rtx=rtx,
                    chars=len(text),
                    text=text,
                ))
        return measurements, metadata

    if variant.tool == "mlx-audio":
        wrapper, startup_s = load_mlx_audio_wrapper(str(variant.params["model_name"]))
        metadata["startup_s"] = startup_s
        for repeat in range(1, repeats + 1):
            for audio_path in audio_files:
                eprint(f"[benchmark] {variant.id} :: {audio_path.name} :: repeat {repeat}/{repeats}")
                text, wall_s, rtx = run_mlx_audio(audio_path, wrapper)
                duration_s = audio_duration_seconds(audio_path)
                save_text(save_text_dir, variant.id, audio_path, repeat, text)
                measurements.append(Measurement(
                    variant_id=variant.id,
                    tool=variant.tool_label,
                    audio=str(audio_path),
                    repeat=repeat,
                    duration_s=duration_s,
                    wall_s=wall_s,
                    rtx=rtx,
                    chars=len(text),
                    text=text,
                    startup_s=startup_s,
                ))
        return measurements, metadata

    if variant.tool == "qwen-asr":
        for repeat in range(1, repeats + 1):
            for audio_path in audio_files:
                eprint(f"[benchmark] {variant.id} :: {audio_path.name} :: repeat {repeat}/{repeats}")
                text, wall_s, rtx = run_qwen(audio_path, Path(variant.params["model_dir"]), list(variant.params["extra_args"]))
                duration_s = audio_duration_seconds(audio_path)
                save_text(save_text_dir, variant.id, audio_path, repeat, text)
                measurements.append(Measurement(
                    variant_id=variant.id,
                    tool=variant.tool_label,
                    audio=str(audio_path),
                    repeat=repeat,
                    duration_s=duration_s,
                    wall_s=wall_s,
                    rtx=rtx,
                    chars=len(text),
                    text=text,
                ))
        return measurements, metadata

    raise RuntimeError(f"unsupported tool: {variant.tool}")


def aggregate_measurements(measurements: list[Measurement]) -> tuple[list[Aggregate], list[VariantSummary]]:
    grouped: dict[tuple[str, str], list[Measurement]] = {}
    for item in measurements:
        grouped.setdefault((item.variant_id, item.audio), []).append(item)

    aggregates: list[Aggregate] = []
    for (variant_id, audio), items in sorted(grouped.items()):
        items.sort(key=lambda entry: entry.repeat)
        total_duration = sum(item.duration_s for item in items)
        total_wall = sum(item.wall_s for item in items)
        aggregates.append(Aggregate(
            variant_id=variant_id,
            tool=items[0].tool,
            audio=audio,
            repeats=len(items),
            duration_s=items[0].duration_s,
            startup_s=items[0].startup_s,
            mean_wall_s=statistics.mean(item.wall_s for item in items),
            min_wall_s=min(item.wall_s for item in items),
            max_wall_s=max(item.wall_s for item in items),
            weighted_rtx=total_duration / max(total_wall, 1e-9),
            chars=items[0].chars,
            text=items[0].text,
        ))

    variant_grouped: dict[str, list[Measurement]] = {}
    for item in measurements:
        variant_grouped.setdefault(item.variant_id, []).append(item)

    summaries: list[VariantSummary] = []
    for variant_id, items in sorted(variant_grouped.items()):
        total_duration = sum(item.duration_s for item in items)
        total_wall = sum(item.wall_s for item in items)
        summaries.append(VariantSummary(
            variant_id=variant_id,
            tool=items[0].tool,
            cases=len({item.audio for item in items}),
            repeats=max(item.repeat for item in items),
            total_audio_s=total_duration,
            total_wall_s=total_wall,
            weighted_rtx=total_duration / max(total_wall, 1e-9),
            startup_s=items[0].startup_s,
        ))

    return aggregates, summaries


def build_text_compares(aggregates: list[Aggregate]) -> list[TextCompare]:
    by_audio: dict[str, list[Aggregate]] = {}
    for item in aggregates:
        by_audio.setdefault(item.audio, []).append(item)

    compares: list[TextCompare] = []
    for audio, rows in sorted(by_audio.items()):
        rows.sort(key=lambda row: row.variant_id)
        for lhs, rhs in itertools.combinations(rows, 2):
            lhs_norm = normalize_text(lhs.text)
            rhs_norm = normalize_text(rhs.text)
            compares.append(TextCompare(
                audio=audio,
                lhs=lhs.variant_id,
                rhs=rhs.variant_id,
                raw_exact=lhs.text == rhs.text,
                normalized_exact=lhs_norm == rhs_norm,
                cer=char_error_rate(lhs_norm, rhs_norm),
                wer=word_error_rate(lhs_norm, rhs_norm),
                lhs_chars=len(lhs.text),
                rhs_chars=len(rhs.text),
            ))
    return compares


def print_report(aggregates: list[Aggregate], summaries: list[VariantSummary], compares: list[TextCompare], compare_text: bool) -> None:
    print("\nPer-audio results\n")
    print("| Run | Audio | Repeats | Startup s | Mean wall s | RTx | Chars |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in aggregates:
        startup = "" if row.startup_s is None else f"{row.startup_s:.2f}"
        print(
            f"| {row.variant_id} | {Path(row.audio).name} | {row.repeats} | {startup} | {row.mean_wall_s:.3f} | {row.weighted_rtx:.2f} | {row.chars} |"
        )

    print("\nTotals by run\n")
    print("| Run | Cases | Repeats | Startup s | Total audio s | Total wall s | Weighted RTx |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        startup = "" if row.startup_s is None else f"{row.startup_s:.2f}"
        print(
            f"| {row.variant_id} | {row.cases} | {row.repeats} | {startup} | {row.total_audio_s:.1f} | {row.total_wall_s:.3f} | {row.weighted_rtx:.2f} |"
        )

    if compare_text and compares:
        print("\nTranscript comparisons\n")
        print("| Audio | Left | Right | Raw exact | Normalized exact | CER | WER | Chars |")
        print("|---|---|---|:---:|:---:|---:|---:|---:|")
        for item in compares:
            print(
                f"| {Path(item.audio).name} | {item.lhs} | {item.rhs} | {'yes' if item.raw_exact else 'no'} | {'yes' if item.normalized_exact else 'no'} | {item.cer:.4f} | {item.wer:.4f} | {item.lhs_chars} / {item.rhs_chars} |"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composable batch transcription benchmark")
    parser.add_argument("--audio", type=Path, action="append", help="Audio file to benchmark. Repeat for multiple files.")
    parser.add_argument("--tool", action="append", choices=["yuwp", "yuwp-cli", "mlx-audio", "qwen-asr"], help="Tool to benchmark. Repeat for multiple tools.")
    parser.add_argument("--repeat", type=int, default=1, help="Run each audio / variant combination N times.")
    parser.add_argument("--compare-text", action="store_true", help="Compare transcripts pairwise for runs that share the same audio.")
    parser.add_argument("--save-text-dir", type=Path, default=None, help="Save transcript text files for each run to this directory.")
    parser.add_argument("--json", default=None, help="Write full JSON output to this path, or '-' for stdout.")
    parser.add_argument("--compact", action="store_true", help="When writing JSON, omit indentation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned benchmark matrix without executing anything.")

    parser.add_argument("--yuwp-model", type=Path, action="append", help=f"Yuwp model directory. Repeat for multiple model variants. Default: {DEFAULT_YUWP_MODEL}")
    parser.add_argument("--yuwp-chunking", action="append", choices=["vad", "energy"], help="Yuwp server chunking mode. Repeat to compare VAD vs low-energy fallback.")

    parser.add_argument("--mlx-model", action="append", help=f"mlx-audio model name. Repeat for multiple models. Default: {DEFAULT_MLX_MODEL}")

    parser.add_argument("--qwen-model", type=Path, action="append", help=f"qwen_asr model directory. Repeat for multiple models. Default: {DEFAULT_QWEN_MODEL}")
    parser.add_argument("--qwen-args", action="append", help="Extra qwen_asr CLI args as a single shell string, e.g. '-S 30 -W 3'. Repeat for multiple qwen variants.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_files, variants = validate_args(args)
    plan_text = describe_plan(audio_files, variants, repeats=args.repeat)

    if args.dry_run:
        print(plan_text)
        return 0

    eprint(plan_text)
    all_measurements: list[Measurement] = []
    metadata: dict[str, dict[str, Any]] = {}

    for variant in variants:
        variant_measurements, variant_metadata = run_variant(variant, audio_files, args.repeat, args.save_text_dir)
        all_measurements.extend(variant_measurements)
        if variant_metadata:
            metadata[variant.id] = variant_metadata

    aggregates, summaries = aggregate_measurements(all_measurements)
    compares = build_text_compares(aggregates) if args.compare_text else []
    print_report(aggregates, summaries, compares, compare_text=args.compare_text)

    if args.json is not None:
        payload = {
            "fetched_at": iso_now(),
            "config": {
                "audio": [str(path) for path in audio_files],
                "tools": args.tool,
                "repeat": args.repeat,
                "compare_text": args.compare_text,
                "save_text_dir": str(args.save_text_dir) if args.save_text_dir else None,
                "yuwp_model": [str(path) for path in (args.yuwp_model or [DEFAULT_YUWP_MODEL])],
                "yuwp_chunking": args.yuwp_chunking or ["vad"],
                "mlx_model": args.mlx_model or [DEFAULT_MLX_MODEL],
                "qwen_model": [str(path) for path in (args.qwen_model or [DEFAULT_QWEN_MODEL])],
                "qwen_args": args.qwen_args or [""],
            },
            "variants": [
                {
                    "id": variant.id,
                    "tool": variant.tool,
                    "tool_label": variant.tool_label,
                    "params": {key: str(value) if isinstance(value, Path) else value for key, value in variant.params.items()},
                    "metadata": metadata.get(variant.id, {}),
                }
                for variant in variants
            ],
            "measurements": [asdict(item) for item in all_measurements],
            "aggregates": [asdict(item) for item in aggregates],
            "comparisons": [asdict(item) for item in compares],
            "summary": {
                "variant_count": len(variants),
                "audio_count": len(audio_files),
                "total_runs": len(all_measurements),
                "total_audio_s": sum(item.duration_s for item in all_measurements),
            },
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2) + "\n"
        if args.json == "-":
            sys.stdout.write(body)
        else:
            Path(args.json).write_text(body)
            print(f"\nSaved JSON → {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
