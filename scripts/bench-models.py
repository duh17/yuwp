#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy", "requests", "psutil"]
# ///
"""
Benchmark and compare ASR models on the Yuwp sidecar.

For each model: starts the sidecar, measures load time and memory, feeds
test audio through the streaming HTTP API, collects per-chunk latency and
final transcription text, then kills the sidecar and moves to the next model.

Metrics per model:
  - Load time (seconds to HTTP ready)
  - Memory RSS (MB after model load)
  - Per-chunk latency (mean, p50, p95)
  - Realtime factor (streaming time / audio duration)
  - Final transcription text per file

Available models (mlx-community):
  Qwen3-ASR-1.7B-bf16  (default, ~8GB)
  Qwen3-ASR-1.7B-8bit
  Qwen3-ASR-1.7B-4bit
  Qwen3-ASR-0.6B-bf16
  Qwen3-ASR-0.6B-8bit
  Qwen3-ASR-0.6B-4bit

Usage:
  # Compare two models on last 5 dictation files
  uv run scripts/bench-models.py \\
    mlx-community/Qwen3-ASR-1.7B-bf16 \\
    mlx-community/Qwen3-ASR-0.6B-4bit \\
    --last 5

  # Short names work (resolved to mlx-community/Qwen3-ASR-...)
  uv run scripts/bench-models.py 1.7B-bf16 0.6B-4bit --last 5

  # Compare on specific files
  uv run scripts/bench-models.py 1.7B-bf16 0.6B-8bit \\
    --files recording1.wav recording2.wav

  # Single model benchmark
  uv run scripts/bench-models.py 0.6B-8bit --last 10

  # Use a pre-running sidecar (skip start/stop)
  uv run scripts/bench-models.py 1.7B-bf16 --external --port 9748

  # Output METRIC lines for autoresearch
  uv run scripts/bench-models.py 1.7B-bf16 0.6B-4bit --last 5 --metrics

  # Save full results as JSON
  uv run scripts/bench-models.py 1.7B-bf16 0.6B-4bit --last 5 --json results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import requests


SAMPLE_RATE = 16000
CHUNK_SEC = 2.0
CHUNK_SAMPLES = int(CHUNK_SEC * SAMPLE_RATE)
DEFAULT_PORT = 9748
SIDECAR_SCRIPT = Path(__file__).resolve().parent.parent / "Sources" / "sidecar" / "transcribe.py"
READY_TIMEOUT = 180  # seconds — first run may download the model
READY_POLL_INTERVAL = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_name(name: str) -> str:
    """Expand short names like '0.6B-4bit' to full HuggingFace IDs."""
    if "/" in name:
        return name
    return f"mlx-community/Qwen3-ASR-{name}"


def model_short_name(name: str) -> str:
    """Extract short label for display: 'mlx-community/Qwen3-ASR-1.7B-bf16' -> '1.7B-bf16'."""
    return name.split("/")[-1].replace("Qwen3-ASR-", "")


def load_audio_as_pcm(path: str) -> np.ndarray:
    """Convert any audio file to 16-bit PCM at 16kHz mono via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {result.stderr.decode()[:200]}")
    return np.frombuffer(result.stdout, dtype=np.int16)


def audio_duration(path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-i", path, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return 0.0
    return float(result.stdout.strip())


def count_cjk(text: str) -> int:
    """Count CJK characters."""
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')


def percentile(values: list[float], p: float) -> float:
    """Compute percentile from sorted values."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


# ---------------------------------------------------------------------------
# Sidecar lifecycle
# ---------------------------------------------------------------------------

def start_sidecar(model: str, port: int) -> subprocess.Popen:
    """Launch the sidecar process in serve-only mode. Returns the Popen handle."""
    cmd = ["uv", "run", "--script", str(SIDECAR_SCRIPT), model, "--serve-only", "--port", str(port)]
    print(f"  Starting sidecar: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def wait_for_ready(port: int, timeout: float = READY_TIMEOUT) -> float:
    """Poll the sidecar /v1/info endpoint until ready. Returns load time in seconds."""
    url = f"http://127.0.0.1:{port}/v1/info"
    t0 = time.time()
    deadline = t0 + timeout

    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.ok and resp.json().get("status") == "ready":
                return time.time() - t0
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(READY_POLL_INTERVAL)

    raise TimeoutError(f"Sidecar not ready after {timeout}s on port {port}")


def stop_sidecar(proc: subprocess.Popen) -> None:
    """Gracefully stop the sidecar process."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def measure_memory(proc: subprocess.Popen) -> float:
    """Measure total RSS of the sidecar process tree in MB."""
    try:
        parent = psutil.Process(proc.pid)
        total = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


# ---------------------------------------------------------------------------
# Streaming transcription via HTTP
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    file: str
    duration_s: float
    total_time_s: float
    chunk_latencies_ms: list[float]
    text: str
    stream_text: str  # streaming-only (before batch correction)
    batch_text: str   # batch-corrected (from session stop)
    saved_text: str   # saved transcript from companion JSON (if available)
    cjk_count: int

    @property
    def realtime_factor(self) -> float:
        return self.total_time_s / self.duration_s if self.duration_s > 0 else 0

    @property
    def avg_chunk_ms(self) -> float:
        return sum(self.chunk_latencies_ms) / len(self.chunk_latencies_ms) if self.chunk_latencies_ms else 0


@dataclass
class ModelResult:
    model: str
    short_name: str
    load_time_s: float
    memory_rss_mb: float
    file_results: list[FileResult] = field(default_factory=list)

    @property
    def all_chunk_latencies(self) -> list[float]:
        out: list[float] = []
        for fr in self.file_results:
            out.extend(fr.chunk_latencies_ms)
        return out

    @property
    def avg_chunk_ms(self) -> float:
        lats = self.all_chunk_latencies
        return sum(lats) / len(lats) if lats else 0

    @property
    def p50_chunk_ms(self) -> float:
        return percentile(self.all_chunk_latencies, 50)

    @property
    def p95_chunk_ms(self) -> float:
        return percentile(self.all_chunk_latencies, 95)

    @property
    def total_audio_s(self) -> float:
        return sum(fr.duration_s for fr in self.file_results)

    @property
    def total_stream_s(self) -> float:
        return sum(fr.total_time_s for fr in self.file_results)

    @property
    def avg_realtime_factor(self) -> float:
        ta = self.total_audio_s
        return self.total_stream_s / ta if ta > 0 else 0


def stream_file(path: str, port: int) -> FileResult:
    """Stream one audio file through the sidecar and collect metrics."""
    pcm = load_audio_as_pcm(path)
    duration_s = len(pcm) / SAMPLE_RATE
    base_url = f"http://127.0.0.1:{port}"
    chunk_latencies: list[float] = []

    # Create session
    t_total = time.time()
    resp = requests.post(f"{base_url}/v1/audio/transcriptions/stream", json={})
    resp.raise_for_status()
    sid = resp.json()["session_id"]

    # Feed chunks
    stream_text = ""
    for i in range(0, len(pcm), CHUNK_SAMPLES):
        chunk = pcm[i:i + CHUNK_SAMPLES]
        t_chunk = time.time()
        resp = requests.post(
            f"{base_url}/v1/audio/transcriptions/stream/{sid}",
            data=chunk.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        chunk_ms = (time.time() - t_chunk) * 1000
        chunk_latencies.append(chunk_ms)
        if resp.ok:
            data = resp.json()
            stream_text = data.get("text", stream_text)

    # Stop session — get batch-corrected final text
    batch_text = stream_text
    resp = requests.delete(f"{base_url}/v1/audio/transcriptions/stream/{sid}")
    if resp.ok:
        batch_text = resp.json().get("text", batch_text)

    # Load saved transcript from companion JSON if available
    saved_text = ""
    json_path = path.replace(".flac", ".json").replace(".wav", ".json")
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                meta = json.load(f)
            saved_text = meta.get("transcript", "")
        except (json.JSONDecodeError, OSError):
            pass

    total_time = time.time() - t_total
    final_text = batch_text or stream_text

    return FileResult(
        file=os.path.basename(path),
        duration_s=duration_s,
        total_time_s=total_time,
        chunk_latencies_ms=chunk_latencies,
        text=final_text,
        stream_text=stream_text,
        batch_text=batch_text,
        saved_text=saved_text,
        cjk_count=count_cjk(final_text),
    )


# ---------------------------------------------------------------------------
# Benchmark a single model
# ---------------------------------------------------------------------------

def benchmark_model(model: str, files: list[str], port: int, external: bool) -> ModelResult | None:
    """Run the full benchmark for one model. Returns None if the model fails to load."""
    short = model_short_name(model)
    print(f"\n{'='*70}")
    print(f"  Model: {model}")
    print(f"{'='*70}")

    proc: subprocess.Popen | None = None
    load_time = 0.0
    memory_mb = 0.0

    if external:
        # Assume sidecar is already running
        print("  Using external sidecar")
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/v1/info", timeout=5)
            if not resp.ok:
                print(f"  ERROR: external sidecar not responding on port {port}")
                return None
            info = resp.json()
            print(f"  Connected — model: {info.get('model', 'unknown')}")
        except requests.ConnectionError:
            print(f"  ERROR: no sidecar on port {port}")
            return None
    else:
        proc = start_sidecar(model, port)
        try:
            load_time = wait_for_ready(port)
            print(f"  Ready in {load_time:.1f}s")
        except TimeoutError as e:
            print(f"  ERROR: {e}")
            # Dump stderr for debugging
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                if stderr:
                    print(f"  Sidecar stderr:\n{stderr[:500]}")
            stop_sidecar(proc)
            return None

        memory_mb = measure_memory(proc)
        print(f"  Memory RSS: {memory_mb:.0f} MB")

    result = ModelResult(
        model=model,
        short_name=short,
        load_time_s=load_time,
        memory_rss_mb=memory_mb,
    )

    # Run each file
    for i, path in enumerate(files):
        fname = os.path.basename(path)
        dur = audio_duration(path)
        print(f"  [{i+1}/{len(files)}] {fname} ({dur:.1f}s)...", end=" ", flush=True)

        try:
            fr = stream_file(path, port)
            result.file_results.append(fr)
            rt = fr.realtime_factor
            avg = fr.avg_chunk_ms
            print(f"{fr.total_time_s:.2f}s  (avg chunk: {avg:.0f}ms, {rt:.3f}x RT)")
        except Exception as e:
            print(f"ERROR: {e}")

    # Cleanup
    if proc is not None:
        stop_sidecar(proc)
        print("  Sidecar stopped")

    return result


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_audio_files(args: argparse.Namespace) -> list[str]:
    """Resolve audio file paths from CLI args."""
    if args.files:
        expanded = []
        for f in args.files:
            p = os.path.expanduser(f)
            if os.path.isfile(p):
                expanded.append(p)
            else:
                print(f"WARNING: file not found: {f}")
        return expanded

    # Search for recordings in the default Yuwp directory
    yuwp_dir = os.path.expanduser("~/Library/Application Support/Yuwp/recordings/")

    candidates: list[str] = []
    candidates.extend(sorted(glob.glob(os.path.join(yuwp_dir, "*.wav"))))
    candidates.extend(sorted(glob.glob(os.path.join(yuwp_dir, "**/*.flac"), recursive=True)))

    if args.date:
        date_str = args.date.replace("-", "")
        filtered = [f for f in candidates if args.date in f or date_str in f]
        return filtered

    # Default: last N files (sorted by path/date)
    n = args.last or 5
    return candidates[-n:]


# ---------------------------------------------------------------------------
# Output: comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: list[ModelResult]) -> None:
    """Print a side-by-side comparison table."""
    if not results:
        return

    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")

    # Column widths
    label_w = 28
    col_w = max(14, max(len(r.short_name) for r in results) + 2)

    # Header
    header = f"{'':>{label_w}}"
    for r in results:
        header += f"  {r.short_name:>{col_w}}"
    print(header)
    print("-" * len(header))

    # Rows
    rows: list[tuple[str, list[str]]] = [
        ("Load time (s)", [f"{r.load_time_s:.1f}" for r in results]),
        ("Memory RSS (MB)", [f"{r.memory_rss_mb:.0f}" for r in results]),
        ("Files tested", [f"{len(r.file_results)}" for r in results]),
        ("Total audio (s)", [f"{r.total_audio_s:.1f}" for r in results]),
        ("Total stream time (s)", [f"{r.total_stream_s:.1f}" for r in results]),
        ("Avg chunk latency (ms)", [f"{r.avg_chunk_ms:.0f}" for r in results]),
        ("P50 chunk latency (ms)", [f"{r.p50_chunk_ms:.0f}" for r in results]),
        ("P95 chunk latency (ms)", [f"{r.p95_chunk_ms:.0f}" for r in results]),
        ("Realtime factor", [f"{r.avg_realtime_factor:.4f}" for r in results]),
    ]

    for label, vals in rows:
        line = f"{label:>{label_w}}"
        for v in vals:
            line += f"  {v:>{col_w}}"
        print(line)


def print_text_comparison(results: list[ModelResult]) -> None:
    """Print per-file text comparison across models, showing streaming vs batch."""
    # Collect all filenames
    all_files: list[str] = []
    seen: set[str] = set()
    for r in results:
        for fr in r.file_results:
            if fr.file not in seen:
                all_files.append(fr.file)
                seen.add(fr.file)

    print(f"\n{'='*70}")
    print("TEXT COMPARISON (streaming vs batch vs saved)")
    print(f"{'='*70}")

    diffs_found = 0
    for fname in all_files:
        file_results: list[tuple[str, FileResult]] = []
        for r in results:
            for fr in r.file_results:
                if fr.file == fname:
                    file_results.append((r.short_name, fr))
                    break

        if not file_results:
            continue

        # Check if there are any interesting differences
        all_texts: list[str] = []
        for name, fr in file_results:
            all_texts.extend([fr.stream_text, fr.batch_text])
        if file_results[0][1].saved_text:
            all_texts.append(file_results[0][1].saved_text)

        stripped = [t.strip() for t in all_texts if t]
        if len(set(stripped)) <= 1:
            continue

        diffs_found += 1
        dur = file_results[0][1].duration_s
        print(f"\n  {fname} ({dur:.1f}s):")

        # Show saved transcript first if available
        saved = file_results[0][1].saved_text
        if saved:
            preview = saved[:120].replace("\n", " ")
            cjk = count_cjk(saved)
            cjk_note = f" [CJK: {cjk}]" if cjk > 0 else ""
            print(f"    {'saved':>20}: ({len(saved)} chars{cjk_note}) {preview}{'...' if len(saved) > 120 else ''}")

        for name, fr in file_results:
            # Streaming text
            s_preview = fr.stream_text[:120].replace("\n", " ")
            s_cjk = count_cjk(fr.stream_text)
            s_note = f" [CJK: {s_cjk}]" if s_cjk > 0 else ""
            print(f"    {name + ' stream':>20}: ({len(fr.stream_text)} chars{s_note}) {s_preview}{'...' if len(fr.stream_text) > 120 else ''}")

            # Batch text (only if different from streaming)
            if fr.batch_text.strip() != fr.stream_text.strip():
                b_preview = fr.batch_text[:120].replace("\n", " ")
                b_cjk = count_cjk(fr.batch_text)
                b_note = f" [CJK: {b_cjk}]" if b_cjk > 0 else ""
                print(f"    {name + ' batch':>20}: ({len(fr.batch_text)} chars{b_note}) {b_preview}{'...' if len(fr.batch_text) > 120 else ''}")

    if diffs_found == 0:
        print("  All transcriptions identical across models.")
    else:
        print(f"\n  {diffs_found} file(s) with differences")


def print_metrics(results: list[ModelResult]) -> None:
    """Print METRIC lines for autoresearch."""
    for r in results:
        tag = r.short_name.replace("-", "_").replace(".", "")
        print(f"METRIC {tag}_load_s={r.load_time_s:.1f}")
        print(f"METRIC {tag}_memory_mb={r.memory_rss_mb:.0f}")
        print(f"METRIC {tag}_avg_chunk_ms={r.avg_chunk_ms:.0f}")
        print(f"METRIC {tag}_p50_chunk_ms={r.p50_chunk_ms:.0f}")
        print(f"METRIC {tag}_p95_chunk_ms={r.p95_chunk_ms:.0f}")
        print(f"METRIC {tag}_realtime_factor={r.avg_realtime_factor:.4f}")
        print(f"METRIC {tag}_files={len(r.file_results)}")
        print(f"METRIC {tag}_total_audio_s={r.total_audio_s:.1f}")


def save_json(results: list[ModelResult], path: str) -> None:
    """Save full results as JSON."""
    out = []
    for r in results:
        entry: dict[str, Any] = {
            "model": r.model,
            "short_name": r.short_name,
            "load_time_s": round(r.load_time_s, 2),
            "memory_rss_mb": round(r.memory_rss_mb, 1),
            "avg_chunk_ms": round(r.avg_chunk_ms, 1),
            "p50_chunk_ms": round(r.p50_chunk_ms, 1),
            "p95_chunk_ms": round(r.p95_chunk_ms, 1),
            "avg_realtime_factor": round(r.avg_realtime_factor, 5),
            "files": [],
        }
        for fr in r.file_results:
            entry["files"].append({
                "file": fr.file,
                "duration_s": round(fr.duration_s, 1),
                "total_time_s": round(fr.total_time_s, 2),
                "avg_chunk_ms": round(fr.avg_chunk_ms, 1),
                "realtime_factor": round(fr.realtime_factor, 4),
                "cjk_count": fr.cjk_count,
                "stream_text": fr.stream_text,
                "batch_text": fr.batch_text,
                "saved_text": fr.saved_text,
                "chunk_latencies_ms": [round(l, 1) for l in fr.chunk_latencies_ms],
            })
        out.append(entry)

    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ASR models on the Yuwp sidecar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 1.7B-bf16 0.6B-4bit --last 5
  %(prog)s 0.6B-8bit --last 10
  %(prog)s 1.7B-bf16 --external --port 9748
  %(prog)s 1.7B-bf16 0.6B-4bit --last 5 --metrics
""",
    )
    parser.add_argument("models", nargs="+", help="Model names (short: '0.6B-4bit' or full HF ID)")
    parser.add_argument("--files", nargs="+", help="Specific audio files to test")
    parser.add_argument("--last", type=int, default=None, help="Test last N audio files (default: 5)")
    parser.add_argument("--date", type=str, default=None, help="Filter files by date (YYYY-MM-DD)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Sidecar port (default: {DEFAULT_PORT})")
    parser.add_argument("--external", action="store_true", help="Use pre-running sidecar (skip start/stop)")
    parser.add_argument("--metrics", action="store_true", help="Output METRIC lines for autoresearch")
    parser.add_argument("--json", type=str, default=None, metavar="PATH", help="Save full results as JSON")
    args = parser.parse_args()

    models = [resolve_model_name(m) for m in args.models]
    files = find_audio_files(args)

    if not files:
        print("No audio files found. Use --files, --last, or --date.")
        sys.exit(1)

    print(f"Benchmarking {len(models)} model(s) on {len(files)} file(s)")
    for m in models:
        print(f"  - {m}")
    print()

    results: list[ModelResult] = []
    for model in models:
        r = benchmark_model(model, files, args.port, args.external)
        if r is not None:
            results.append(r)

    if not results:
        print("\nNo models completed successfully.")
        sys.exit(1)

    # Output
    print_comparison(results)
    print_text_comparison(results)

    if args.metrics:
        print()
        print_metrics(results)

    if args.json:
        save_json(results, args.json)


if __name__ == "__main__":
    main()
