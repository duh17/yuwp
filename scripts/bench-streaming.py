#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "mlx-audio>=0.2.2",
#   "mlx-lm",
#   "numpy",
#   "soundfile",
#   "scipy",
# ]
# ///
"""Benchmark: Python sidecar streaming vs Native Swift streaming.

Feeds the same WAV files through both engines with identical config,
measures per-chunk latency and total time.

Usage: bench-streaming.py <wav-file>... [--model <name>] [--chunk-sec 2.0] [--rounds 3]
"""

import sys
import os
import time
import json
import subprocess
import argparse
import numpy as np
import soundfile as sf

# Add sidecar to path so we can import its streaming logic
SIDECAR_DIR = os.path.join(os.path.dirname(__file__), "..", "Sources", "sidecar")
sys.path.insert(0, SIDECAR_DIR)

SAMPLE_RATE = 16000


def bench_python(wav_path: str, model, chunk_sec: float) -> dict:
    """Run Python sidecar streaming on a WAV file."""
    import mlx.core as mx
    from transcribe import StreamSession, StreamConfig, process_chunk

    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr)).astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    config = StreamConfig(chunk_sec=chunk_sec)
    session = StreamSession(model=model, config=config)

    mem_before = mx.metal.get_active_memory()
    chunk_times = []
    offset = 0
    t0 = time.time()

    while offset < len(audio):
        end = min(offset + chunk_samples, len(audio))
        chunk = audio[offset:end]
        ct0 = time.time()
        result = process_chunk(session, chunk)
        chunk_times.append((time.time() - ct0) * 1000)
        offset = end

    total_ms = (time.time() - t0) * 1000
    audio_dur = len(audio) / SAMPLE_RATE
    mem_after = mx.metal.get_active_memory()
    mem_peak = mx.metal.get_peak_memory()

    return {
        "text": result.get("text", "") if isinstance(result, dict) else "",
        "total_ms": total_ms,
        "audio_sec": audio_dur,
        "chunks": len(chunk_times),
        "chunk_times_ms": chunk_times,
        "mean_chunk_ms": sum(chunk_times) / len(chunk_times) if chunk_times else 0,
        "p50_chunk_ms": sorted(chunk_times)[len(chunk_times) // 2] if chunk_times else 0,
        "p95_chunk_ms": sorted(chunk_times)[int(len(chunk_times) * 0.95)] if chunk_times else 0,
        "mem_mb": mem_after // (1024 * 1024),
        "peak_mb": mem_peak // (1024 * 1024),
        "delta_mb": (mem_after - mem_before) // (1024 * 1024),
    }


def bench_native(wav_path: str, model_dir: str, chunk_sec: float) -> dict:
    """Run native Swift streaming on a WAV file via CLI."""
    binary = os.path.join(
        os.path.dirname(__file__), "..", ".build", "arm64-apple-macosx", "release", "asr-stream-test"
    )
    if not os.path.exists(binary):
        print(f"ERROR: Native binary not found at {binary}", file=sys.stderr)
        return {}

    cmd = [binary, wav_path, model_dir, "--chunk-sec", str(chunk_sec),
           "--no-batch", "--no-batch-retranscribe"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Parse stderr for chunk timings
    chunk_times = []
    total_ms = 0
    audio_sec = 0
    text = result.stdout.strip()

    for line in result.stderr.split("\n"):
        if "Audio:" in line and "samples" in line:
            # [stream-test] Audio: 11.00s, 176000 samples
            parts = line.split()
            for p in parts:
                if p.endswith("s,"):
                    audio_sec = float(p.rstrip("s,"))
        if line.strip().startswith("chunk"):
            # chunk  1 (2.0s): 84ms (enc=25 pfx=22 dec=34 reuse=0%)
            try:
                after_colon = line.split(":")[1].strip()
                ms_str = after_colon.split("ms")[0].strip()
                chunk_times.append(float(ms_str))
            except (IndexError, ValueError):
                pass
        if "Streaming done in" in line:
            try:
                # "[stream-test] Streaming done in 1.14s"
                idx = line.index("done in ") + len("done in ")
                total_ms = float(line[idx:].strip().rstrip("s")) * 1000
            except (IndexError, ValueError):
                pass

    mem_mb = 0
    peak_mb = 0
    for line in result.stderr.split("\n"):
        if "Memory:" in line:
            for part in line.split():
                if part.startswith("active="):
                    mem_mb = int(part.split("=")[1].rstrip("MB"))
                if part.startswith("peak="):
                    peak_mb = int(part.split("=")[1].rstrip("MB"))

    return {
        "text": text,
        "total_ms": total_ms,
        "audio_sec": audio_sec,
        "chunks": len(chunk_times),
        "chunk_times_ms": chunk_times,
        "mean_chunk_ms": sum(chunk_times) / len(chunk_times) if chunk_times else 0,
        "p50_chunk_ms": sorted(chunk_times)[len(chunk_times) // 2] if chunk_times else 0,
        "p95_chunk_ms": sorted(chunk_times)[int(len(chunk_times) * 0.95)] if chunk_times else 0,
        "mem_mb": mem_mb,
        "peak_mb": peak_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Streaming ASR benchmark")
    parser.add_argument("wavs", nargs="+", help="WAV files to benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen3-ASR-1.7B-bf16",
                        help="Model name for Python (default: 1.7B-bf16)")
    parser.add_argument("--model-dir", default=None,
                        help="Local model dir for native (auto-detected from --model)")
    parser.add_argument("--chunk-sec", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()

    # Auto-detect model dir for native
    model_dir = args.model_dir
    if not model_dir:
        # Try to find in HF cache
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        model_slug = args.model.replace("/", "--")
        model_cache = os.path.join(hf_cache, f"models--{model_slug}", "snapshots")
        if os.path.exists(model_cache):
            snapshots = [d for d in os.listdir(model_cache)
                         if os.path.isdir(os.path.join(model_cache, d))]
            if snapshots:
                model_dir = os.path.join(model_cache, snapshots[0])

    if not model_dir and not args.python_only:
        print(f"ERROR: Cannot find model dir for {args.model}", file=sys.stderr)
        sys.exit(1)

    # Load Python model once
    py_model = None
    if not args.native_only:
        print(f"Loading Python model: {args.model}", file=sys.stderr)
        t0 = time.time()
        from transcribe import load_models
        py_model = load_models(args.model, batch_enabled=False)
        print(f"Python model loaded in {time.time() - t0:.1f}s\n", file=sys.stderr)

    if not args.python_only:
        print(f"Native model dir: {model_dir}\n", file=sys.stderr)

    # Benchmark each file
    for wav in args.wavs:
        name = os.path.basename(wav)
        print(f"{'='*60}", file=sys.stderr)
        print(f"  {name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        py_results = []
        nat_results = []

        for r in range(args.rounds):
            tag = f"round {r+1}/{args.rounds}"

            if not args.native_only and py_model is not None:
                res = bench_python(wav, py_model, args.chunk_sec)
                py_results.append(res)
                print(f"  Python  {tag}: {res['total_ms']:7.0f}ms total, "
                      f"mean={res['mean_chunk_ms']:.0f}ms p50={res['p50_chunk_ms']:.0f}ms "
                      f"p95={res['p95_chunk_ms']:.0f}ms ({res['chunks']} chunks)", file=sys.stderr)

            if not args.python_only and model_dir:
                res = bench_native(wav, model_dir, args.chunk_sec)
                nat_results.append(res)
                print(f"  Native  {tag}: {res['total_ms']:7.0f}ms total, "
                      f"mean={res['mean_chunk_ms']:.0f}ms p50={res['p50_chunk_ms']:.0f}ms "
                      f"p95={res['p95_chunk_ms']:.0f}ms ({res['chunks']} chunks)", file=sys.stderr)

        # Summary
        print(file=sys.stderr)
        audio_sec = (py_results[0] if py_results else nat_results[0])["audio_sec"]
        print(f"  Audio: {audio_sec:.1f}s, chunk={args.chunk_sec}s", file=sys.stderr)

        if py_results:
            best_py = min(r["total_ms"] for r in py_results)
            avg_chunk = sum(r["mean_chunk_ms"] for r in py_results) / len(py_results)
            mem = py_results[-1].get("peak_mb", 0)
            delta = py_results[-1].get("delta_mb", 0)
            print(f"  Python  best: {best_py:.0f}ms ({audio_sec/best_py*1000:.0f}x RT), "
                  f"avg chunk: {avg_chunk:.0f}ms, peak={mem}MB delta={delta}MB", file=sys.stderr)

        if nat_results:
            best_nat = min(r["total_ms"] for r in nat_results)
            avg_chunk = sum(r["mean_chunk_ms"] for r in nat_results) / len(nat_results)
            mem = nat_results[-1].get("peak_mb", 0)
            print(f"  Native  best: {best_nat:.0f}ms ({audio_sec/best_nat*1000:.0f}x RT), "
                  f"avg chunk: {avg_chunk:.0f}ms, peak={mem}MB", file=sys.stderr)

        if py_results and nat_results:
            best_py = min(r["total_ms"] for r in py_results)
            best_nat = min(r["total_ms"] for r in nat_results)
            diff = (best_nat - best_py) / best_py * 100
            winner = "Native" if best_nat < best_py else "Python"
            print(f"  → {winner} wins by {abs(diff):.1f}%", file=sys.stderr)

        print(file=sys.stderr)


if __name__ == "__main__":
    main()
