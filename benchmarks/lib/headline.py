#!/usr/bin/env python3
"""Headline Yuwp numbers: hot batch, live stdio dictation, CLI transcribe, CLI TTS.

    uv run benchmarks/cli.py asr headline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKET_MS = 100
SAMPLE_RATE = 16000
JFK_ENERGY_ONSET_S = 0.30
TTS_TEXT = "Hello from Yuwp. This is a short streaming speech test."

DEFAULT_ASR = [
    ("0.6B-bf16", Path.home() / "Library/Application Support/Yuwp/models/mlx-community--Qwen3-ASR-0.6B-bf16"),
    ("0.6B-4bit", Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3-ASR-0.6B-4bit/snapshots/313d850181767edf09f00a9c289becca70e58cd0"),
    ("1.7B-bf16", Path.home() / "Library/Application Support/Yuwp/models/mlx-community--Qwen3-ASR-1.7B-bf16"),
    ("1.7B-4bit", Path.home() / "Library/Application Support/Yuwp/models/mlx-community--Qwen3-ASR-1.7B-4bit"),
]
DEFAULT_TTS = [
    ("1.7B-CustomVoice", Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/b611c9f8f2ad5c741ed9c7a0a6a3750e43e0dfd7"),
]


class HeadlineError(RuntimeError):
    pass


def repo_bin(name: str) -> Path:
    for path in (
        REPO / ".build/out/Products/Release" / name,
        REPO / ".build/arm64-apple-macosx/release" / name,
    ):
        if path.exists():
            return path
    raise HeadlineError(f"missing {name}; build with swift build -c release --product {name}")


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def load_pcm(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != SAMPLE_RATE or wav.getsampwidth() != 2:
            raise HeadlineError(f"need 16 kHz mono s16le: {path}")
        pcm = wav.readframes(wav.getnframes())
        return pcm, wav.getnframes() / float(SAMPLE_RATE)


def packets(pcm: bytes) -> list[bytes]:
    size = SAMPLE_RATE * PACKET_MS // 1000 * 2
    return [pcm[i : i + size] for i in range(0, len(pcm), size)]


class StdioClient:
    def __init__(self, process: subprocess.Popen[bytes], timeout_s: float = 120) -> None:
        if process.stdin is None or process.stdout is None:
            raise HeadlineError("stdio pipes missing")
        self.process = process
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.timeout_s = timeout_s
        self.request_id = 1

    def close(self) -> None:
        try:
            self.stdin.close()
        except Exception:
            pass

    def _read_exact(self, size: int, timeout_s: float) -> bytes:
        if size == 0:
            return b""
        fd = self.stdout.fileno()
        chunks: list[bytes] = []
        remaining = size
        deadline = time.perf_counter() + timeout_s
        while remaining > 0:
            wait = deadline - time.perf_counter()
            if wait <= 0:
                raise HeadlineError("stdio read timeout")
            ready, _, _ = select.select([fd], [], [], wait)
            if not ready:
                raise HeadlineError("stdio read timeout")
            chunk = os.read(fd, remaining)
            if not chunk:
                raise HeadlineError("stdio pipe closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def request(self, command: str, session_id: str | None = None, binary: bytes = b"") -> dict:
        if self.process.poll() is not None:
            raise HeadlineError(f"server exited {self.process.returncode}")
        request_id = self.request_id
        self.request_id += 1
        metadata: dict = {"id": request_id, "command": command}
        if session_id is not None:
            metadata["session_id"] = session_id
        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode()
        self.stdin.write(struct.pack(">II", len(metadata_bytes), len(binary)) + metadata_bytes + binary)
        self.stdin.flush()
        header = self._read_exact(8, self.timeout_s)
        metadata_len, binary_len = struct.unpack(">II", header)
        payload = json.loads(self._read_exact(metadata_len, self.timeout_s))
        if binary_len:
            _ = self._read_exact(binary_len, self.timeout_s)
        if payload.get("id") != request_id or not payload.get("ok", False):
            raise HeadlineError(str(payload.get("error", payload)))
        return payload

    def wait_ready(self) -> None:
        if self.request("info").get("status") != "ready":
            raise HeadlineError("server not ready")


def start_stdio(asr_bin: Path, model: Path, log_path: Path) -> subprocess.Popen[bytes]:
    log = open(log_path, "wb")
    return subprocess.Popen(
        [str(asr_bin), "serve", "--model", str(model), "--transport", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        start_new_session=True,
    )


def stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def dictation_1x(client: StdioClient, pcm: bytes) -> dict:
    sid = client.request("create")["session_id"]
    first_text = None
    t_first = None
    t_audio0 = None
    last = ""
    for index, chunk in enumerate(packets(pcm)):
        if t_audio0 is None:
            t_audio0 = time.perf_counter()
        delay = t_audio0 + index * (PACKET_MS / 1000.0) - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        payload = client.request("feed", session_id=sid, binary=chunk)
        text = payload.get("text") or ""
        last = text
        if first_text is None and str(text).strip():
            first_text = text
            t_first = time.perf_counter()
    t_stop0 = time.perf_counter()
    stopped = client.request("stop", session_id=sid)
    t_stop1 = time.perf_counter()
    ttft = None if t_first is None or t_audio0 is None else (t_first - t_audio0) * 1000.0
    return {
        "file_start_ttft_ms": ttft,
        "energy_onset_ttft_ms": None if ttft is None else ttft - JFK_ENERGY_ONSET_S * 1000.0,
        "finalize_ms": (t_stop1 - t_stop0) * 1000.0,
        "first_text": first_text,
        "final_text": stopped.get("text") or last,
    }


def run_batch(bench: Path, model: Path, wav: Path, iterations: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="yuwp-headline-batch-") as tmp:
        shutil.copy(wav, Path(tmp) / wav.name)
        result = subprocess.run(
            [str(bench), str(model), tmp, "--iterations", str(iterations)],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise HeadlineError(result.stderr[-1500:] or f"asr-bench exit {result.returncode}")
    mean = re.search(r"native_inference_mean_s=([0-9.]+)", result.stdout)
    minimum = re.search(r"native_inference_min_s=([0-9.]+)", result.stdout)
    load = re.search(r"native_model_load_s=([0-9.]+)", result.stdout)
    if not mean or not minimum:
        raise HeadlineError(f"asr-bench missing METRIC lines:\n{result.stdout[-500:]}")
    return {
        "mean_s": float(mean.group(1)),
        "min_s": float(minimum.group(1)),
        "load_s": float(load.group(1)) if load else None,
    }


def transcribe_cli(asr_bin: Path, model: Path, wav: Path) -> dict:
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(asr_bin), "transcribe", str(wav), "--model", str(model), "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        raise HeadlineError(result.stderr[-1000:] or f"transcribe exit {result.returncode}")
    payload = json.loads(result.stdout)
    duration = float(payload.get("duration") or 0.0)
    return {"wall_s": wall, "rtf": wall / duration if duration else None, "duration_s": duration}


def tts_cli(tts_bin: Path, model: Path, out: Path) -> dict:
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(tts_bin), "--model", str(model), "--text", TTS_TEXT, "--out", str(out), "--voice", "Ryan", "--language", "English"],
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        raise HeadlineError(result.stderr[-2000:] or f"tts exit {result.returncode}")
    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(out)],
        text=True,
    )
    audio_s = float(probe.strip())
    return {"wall_s": wall, "audio_s": audio_s, "rtf": wall / audio_s if audio_s else None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headline numbers: hot batch (asr-bench), live stdio dictation, CLI transcribe, CLI TTS."
    )
    parser.add_argument("--audio", default=str(REPO / "Tests/fixtures/jfk.wav"))
    parser.add_argument("--json", default=str(REPO / "benchmarks/fixtures/public-jfk.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-iterations", type=int, default=5)
    parser.add_argument("--skip-dictation", action="store_true")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wav = Path(args.audio).resolve()
    asr_bin = repo_bin("yuwp-asr")
    pcm, duration_s = load_pcm(wav)
    report = {
        "audio": {"path": str(wav.relative_to(REPO)) if wav.is_relative_to(REPO) else str(wav), "duration_s": duration_s},
        "asr": [],
        "tts": [],
    }

    for label, model in DEFAULT_ASR:
        if not (model / "model.safetensors").exists():
            print(f"skip {label}", file=sys.stderr)
            continue
        print(f"== {label}", file=sys.stderr)
        row: dict = {"label": label}
        if not args.skip_batch:
            row["batch"] = run_batch(repo_bin("asr-bench"), model, wav, args.batch_iterations)
            print(f"  batch {row['batch']['mean_s']*1000:.0f} ms", file=sys.stderr)
        if not args.skip_dictation:
            proc = start_stdio(asr_bin, model, Path(f"/tmp/yuwp-headline-{label}.log"))
            try:
                client = StdioClient(proc)
                client.wait_ready()
                dictation_1x(client, pcm)
                streams = [dictation_1x(client, pcm) for _ in range(args.repeats)]
                client.close()
            finally:
                stop_proc(proc)
            row["dictation"] = {
                "energy_onset_ttft_ms": summarize(
                    [s["energy_onset_ttft_ms"] for s in streams if s["energy_onset_ttft_ms"] is not None]
                ),
                "finalize_ms": summarize([s["finalize_ms"] for s in streams]),
            }
            print(
                f"  dictation first-text {row['dictation']['energy_onset_ttft_ms']['median']:.0f} ms",
                file=sys.stderr,
            )
        if not args.skip_transcribe:
            transcribe_cli(asr_bin, model, wav)
            times = [transcribe_cli(asr_bin, model, wav) for _ in range(args.repeats)]
            row["transcribe"] = {"wall_s": summarize([t["wall_s"] for t in times])}
            print(f"  transcribe {row['transcribe']['wall_s']['median']:.2f} s", file=sys.stderr)
        report["asr"].append(row)
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")

    if not args.skip_tts:
        tts_bin = repo_bin("yuwp-tts")
        for label, model in DEFAULT_TTS:
            if not (model / "model.safetensors").exists():
                print(f"skip TTS {label}", file=sys.stderr)
                continue
            print(f"== TTS {label}", file=sys.stderr)
            out = Path(f"/tmp/yuwp-headline-tts-{label}.wav")
            tts_cli(tts_bin, model, out)
            timed = [tts_cli(tts_bin, model, out) for _ in range(args.repeats)]
            report["tts"].append(
                {
                    "label": label,
                    "wall_s": summarize([t["wall_s"] for t in timed]),
                    "audio_s": summarize([t["audio_s"] for t in timed]),
                    "rtf": summarize([t["rtf"] for t in timed if t["rtf"] is not None]),
                }
            )
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")

    Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
