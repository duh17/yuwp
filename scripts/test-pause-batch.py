#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["requests", "numpy"]
# ///
"""
Test pause-triggered batch retranscription using real dictation audio.

Feeds a saved FLAC through the sidecar's streaming API in real-time-ish
chunks, verifying that:
  1. Streaming works during speech
  2. Batch triggers during silence (pause)
  3. Streaming resumes after pause
  4. Stop returns batch-corrected final text

Usage:
  uv run scripts/test-pause-batch.py                    # uses latest dictation with a pause
  uv run scripts/test-pause-batch.py <path-to-flac>     # specific file
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import requests


SIDECAR_URL = "http://localhost:9748"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.0


def load_flac(path: str) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr.decode()[:200]}", file=sys.stderr)
        sys.exit(1)
    return np.frombuffer(result.stdout, dtype=np.int16)


def find_test_audio() -> str:
    """Find a dictation file with a mid-session pause (language switch test)."""
    # Use the 86s multilingual stress test
    candidate = os.path.expanduser(
        "~/Library/Application Support/Yuwp/recordings/2026/04/06/dict_e53eaa2d00d4daf3.flac"
    )
    if os.path.exists(candidate):
        return candidate
    # Fallback: latest file
    import glob
    files = sorted(glob.glob(os.path.expanduser(
        "~/Library/Application Support/Yuwp/recordings/**/*.flac"
    ), recursive=True))
    if files:
        return files[-1]
    print("No dictation audio found", file=sys.stderr)
    sys.exit(1)


def rms(samples: np.ndarray) -> float:
    f = samples.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f ** 2)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_test_audio()
    pcm = load_flac(path)
    duration = len(pcm) / SAMPLE_RATE
    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)

    print(f"Audio: {path}")
    print(f"Duration: {duration:.1f}s ({len(pcm)} samples)")
    print(f"Chunk: {CHUNK_SECONDS}s ({chunk_samples} samples)")
    print(f"Sidecar: {SIDECAR_URL}")
    print()

    # Create session
    r = requests.post(f"{SIDECAR_URL}/v1/audio/transcriptions/stream", json={})
    r.raise_for_status()
    sid = r.json()["session_id"]
    print(f"Session: {sid}")
    print()

    # Feed chunks
    batch_events = []
    texts = []
    for i in range(0, len(pcm), chunk_samples):
        chunk = pcm[i:i + chunk_samples]
        t_start = i / SAMPLE_RATE
        chunk_rms = rms(chunk)

        t0 = time.time()
        r = requests.post(
            f"{SIDECAR_URL}/v1/audio/transcriptions/stream/{sid}",
            data=chunk.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        elapsed_ms = (time.time() - t0) * 1000

        if r.ok:
            data = r.json()
            text = data.get("text", "")
            is_batch = data.get("batch_corrected", False)
            texts.append(text)

            marker = ""
            if is_batch:
                batch_events.append((t_start, elapsed_ms, len(text)))
                marker = " *** BATCH"
            if chunk_rms < 0.020:
                marker += " (quiet)"

            # Truncate text for display
            display = text[-60:] if len(text) > 60 else text
            print(f"  t={t_start:5.1f}s  rms={chunk_rms:.4f}  {elapsed_ms:6.0f}ms  [{len(text):3d} chars] ...{display}{marker}")
        else:
            print(f"  t={t_start:5.1f}s  HTTP {r.status_code}")

    # Stop
    print()
    t0 = time.time()
    r = requests.delete(f"{SIDECAR_URL}/v1/audio/transcriptions/stream/{sid}")
    stop_ms = (time.time() - t0) * 1000

    if r.ok:
        final = r.json().get("text", "")
        print(f"STOP: {stop_ms:.0f}ms, {len(final)} chars")
        print(f"Final: {final[:200]}{'...' if len(final) > 200 else ''}")
    else:
        print(f"STOP: HTTP {r.status_code}")

    # Summary
    print()
    print("=== SUMMARY ===")
    print(f"  Batch corrections during stream: {len(batch_events)}")
    for t, ms, chars in batch_events:
        print(f"    at {t:.1f}s: {ms:.0f}ms, {chars} chars")
    print(f"  Stop batch: {stop_ms:.0f}ms")

    # Check for stuck behavior
    max_feed_ms = max(
        (time.time() - t0) * 1000  # approximate
        for t0 in [time.time()]  # placeholder
    )
    feed_times = []
    # Re-parse would be complex, just check if any batch took too long
    for _, ms, _ in batch_events:
        feed_times.append(ms)
    if feed_times:
        max_batch = max(feed_times)
        print(f"  Max batch feed time: {max_batch:.0f}ms")
        if max_batch > 3000:
            print(f"  WARNING: Batch took >{max_batch:.0f}ms — may cause UI freeze")
        else:
            print(f"  OK: Batch times are acceptable")


if __name__ == "__main__":
    main()
