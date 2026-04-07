#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["mlx-audio", "numpy", "requests"]
# ///
"""
Benchmark streaming vs batch ASR accuracy on saved dictation audio.

Compares:
  1. Streaming result (from the sidecar's session API)
  2. Batch result (from mlx-audio's generate_transcription)

Metrics:
  - Character-level diff (shows where streaming diverges from batch)
  - Chinese character preservation (did streaming translate instead of transcribe?)
  - Latency for each mode

Usage:
  # Test on a specific FLAC file
  uv run scripts/bench-streaming-vs-batch.py ~/Library/Application Support/Yuwp/recordings/2026/04/06/dict_e53eaa2d00d4daf3.flac

  # Test on last N dictation files
  uv run scripts/bench-streaming-vs-batch.py --last 5

  # Test all files from a date
  uv run scripts/bench-streaming-vs-batch.py --date 2026-04-06
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import requests


SIDECAR_URL = "http://localhost:9748"
MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
SAMPLE_RATE = 16000


def load_flac_as_pcm(path: str) -> np.ndarray:
    """Load FLAC/WAV file and return 16-bit PCM at 16kHz mono."""
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
    return np.frombuffer(result.stdout, dtype=np.int16)


def streaming_transcribe(pcm: np.ndarray, chunk_duration_s: float = 2.0) -> tuple[str, float]:
    """Transcribe via sidecar streaming session API. Returns (text, elapsed_seconds)."""
    chunk_samples = int(chunk_duration_s * SAMPLE_RATE)

    # Create session
    t0 = time.time()
    res = requests.post(f"{SIDECAR_URL}/v1/audio/transcriptions/stream",
                        json={"stream_config": {"model": MODEL}})
    res.raise_for_status()
    sid = res.json()["session_id"]

    # Feed chunks
    last_text = ""
    for i in range(0, len(pcm), chunk_samples):
        chunk = pcm[i:i + chunk_samples]
        res = requests.post(
            f"{SIDECAR_URL}/v1/audio/transcriptions/stream/{sid}",
            data=chunk.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        if res.ok:
            data = res.json()
            last_text = data.get("text", last_text)

    # Stop session
    res = requests.delete(f"{SIDECAR_URL}/v1/audio/transcriptions/stream/{sid}")
    if res.ok:
        data = res.json()
        last_text = data.get("text", last_text)

    elapsed = time.time() - t0
    return last_text, elapsed


def batch_transcribe(path: str) -> tuple[str, float]:
    """Transcribe via batch mlx-audio. Returns (text, elapsed_seconds)."""
    from mlx_audio.stt.generate import load_model, generate_transcription

    if not hasattr(batch_transcribe, "_model"):
        batch_transcribe._model = load_model(MODEL)

    t0 = time.time()
    result = generate_transcription(batch_transcribe._model, path)
    elapsed = time.time() - t0
    return result.text, elapsed


def count_cjk(text: str) -> int:
    """Count CJK characters in text."""
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')


def char_diff_summary(streaming: str, batch: str) -> str:
    """Simple character-level comparison showing divergence points."""
    # Find first divergence
    min_len = min(len(streaming), len(batch))
    first_diff = min_len
    for i in range(min_len):
        if streaming[i] != batch[i]:
            first_diff = i
            break

    if first_diff == min_len and len(streaming) == len(batch):
        return "  IDENTICAL"

    ctx = 30
    lines = []
    start = max(0, first_diff - ctx)
    lines.append(f"  First divergence at char {first_diff}:")
    lines.append(f"    streaming: ...{streaming[start:first_diff]}|{streaming[first_diff:first_diff+ctx*2]}...")
    lines.append(f"    batch:     ...{batch[start:first_diff]}|{batch[first_diff:first_diff+ctx*2]}...")
    return "\n".join(lines)


def analyze_file(path: str, idx: int = 0, total: int = 1) -> dict:
    """Run both modes on one file and compare."""
    meta_path = path.replace(".flac", ".json").replace(".wav", ".json")
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))

    duration_s = meta.get("durationMs", 0) / 1000
    if duration_s == 0:
        pcm = load_flac_as_pcm(path)
        duration_s = len(pcm) / SAMPLE_RATE

    print(f"\n{'='*70}")
    print(f"[{idx+1}/{total}] {os.path.basename(path)} ({duration_s:.1f}s)")
    print(f"{'='*70}")

    # Streaming
    print("  Streaming...", end=" ", flush=True)
    pcm = load_flac_as_pcm(path)
    stream_text, stream_time = streaming_transcribe(pcm)
    print(f"{stream_time:.2f}s")

    # Batch
    print("  Batch...", end=" ", flush=True)
    batch_text, batch_time = batch_transcribe(path)
    print(f"{batch_time:.2f}s")

    # Compare
    stream_cjk = count_cjk(stream_text)
    batch_cjk = count_cjk(batch_text)

    print(f"\n  Audio duration: {duration_s:.1f}s")
    print(f"  Streaming: {stream_time:.2f}s ({stream_time/duration_s:.3f}x realtime)")
    print(f"  Batch:     {batch_time:.2f}s ({batch_time/duration_s:.3f}x realtime)")
    print(f"  CJK chars: streaming={stream_cjk}, batch={batch_cjk}", end="")
    if batch_cjk > 0 and stream_cjk < batch_cjk:
        lost = batch_cjk - stream_cjk
        print(f"  *** LOST {lost} CJK chars (translated instead of transcribed)")
    else:
        print()

    print(f"\n  Streaming text ({len(stream_text)} chars):")
    print(f"    {stream_text[:200]}{'...' if len(stream_text) > 200 else ''}")
    print(f"\n  Batch text ({len(batch_text)} chars):")
    print(f"    {batch_text[:200]}{'...' if len(batch_text) > 200 else ''}")

    print(f"\n  Diff:")
    print(char_diff_summary(stream_text, batch_text))

    return {
        "file": os.path.basename(path),
        "duration_s": duration_s,
        "stream_time": stream_time,
        "batch_time": batch_time,
        "stream_cjk": stream_cjk,
        "batch_cjk": batch_cjk,
        "cjk_lost": max(0, batch_cjk - stream_cjk),
        "texts_match": stream_text.strip() == batch_text.strip(),
        "stream_text": stream_text,
        "batch_text": batch_text,
    }


def find_audio_files(args) -> list[str]:
    """Resolve audio file paths from CLI args."""
    if args.files:
        return args.files

    base = os.path.expanduser("~/Library/Application Support/Yuwp/recordings/")

    if args.date:
        parts = args.date.split("-")
        date_dir = os.path.join(base, *parts)
        files = sorted(glob.glob(os.path.join(date_dir, "*.flac")))
        if not files:
            print(f"No FLAC files found in {date_dir}")
            sys.exit(1)
        return files

    # --last N: find most recent N files
    all_flacs = sorted(glob.glob(os.path.join(base, "**/*.flac"), recursive=True))
    n = args.last or 3
    return all_flacs[-n:]


def main():
    parser = argparse.ArgumentParser(description="Benchmark streaming vs batch ASR")
    parser.add_argument("files", nargs="*", help="FLAC/WAV files to test")
    parser.add_argument("--last", type=int, default=None, help="Test last N dictation files")
    parser.add_argument("--date", type=str, default=None, help="Test all files from YYYY-MM-DD")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args()

    files = find_audio_files(args)
    if not files:
        print("No audio files found")
        sys.exit(1)

    print(f"Testing {len(files)} file(s) — streaming vs batch")
    print(f"Sidecar: {SIDECAR_URL}")
    print(f"Model: {MODEL}")

    results = []
    for i, f in enumerate(files):
        results.append(analyze_file(f, i, len(files)))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total_dur = sum(r["duration_s"] for r in results)
    total_stream = sum(r["stream_time"] for r in results)
    total_batch = sum(r["batch_time"] for r in results)
    total_cjk_lost = sum(r["cjk_lost"] for r in results)
    matches = sum(1 for r in results if r["texts_match"])

    print(f"  Files tested: {len(results)}")
    print(f"  Total audio: {total_dur:.1f}s")
    print(f"  Streaming total: {total_stream:.1f}s ({total_stream/total_dur:.3f}x realtime)")
    print(f"  Batch total: {total_batch:.1f}s ({total_batch/total_dur:.3f}x realtime)")
    print(f"  Exact matches: {matches}/{len(results)}")
    print(f"  CJK chars lost to translation: {total_cjk_lost}")
    print(f"  Batch-on-stop overhead: +{total_batch:.1f}s for {total_dur:.0f}s of audio")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
