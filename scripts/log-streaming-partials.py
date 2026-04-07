#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy", "requests"]
# ///
"""
Log every streaming partial from the sidecar to analyze rollback volatility.

For each audio file, streams through the HTTP API and captures every partial
text update. Then simulates the TypewriterAnimator logic to show what the
user would actually see — including snap-corrections and large replays.

This reveals the UX pain points: how often does text get replaced mid-display,
how large are the rollback-induced rewrites, and where does the typewriter
animation get interrupted.

Usage:
  # Sidecar must be running on port 9748
  uv run scripts/log-streaming-partials.py --last 5
  uv run scripts/log-streaming-partials.py --files path/to/audio.flac
  uv run scripts/log-streaming-partials.py --last 3 --json results/partials.json

Output per chunk:
  chunk N: "full text so far"
    common_prefix: 42 chars  (stable text that didn't change)
    removed: 5 chars "ello "  (text that got rolled back)
    added: 12 chars "ello world "  (new text appended)
    typewriter: SNAP 5 + ANIMATE 12  (what the animator would do)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests


SAMPLE_RATE = 16000
CHUNK_SEC = 2.0
CHUNK_SAMPLES = int(CHUNK_SEC * SAMPLE_RATE)
SIDECAR_URL = "http://127.0.0.1:9748"


def load_audio_as_pcm(path: str) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
    return np.frombuffer(result.stdout, dtype=np.int16)


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@dataclass
class PartialUpdate:
    chunk_idx: int
    elapsed_ms: float
    text: str
    common_prefix: int      # chars shared with previous update
    removed_chars: int      # chars removed from previous (rollback)
    removed_text: str       # the actual removed text
    added_chars: int        # chars added after common prefix
    added_text: str         # the actual added text
    is_correction: bool     # True if text changed mid-stream (not pure append)
    typewriter_action: str  # SNAP n + ANIMATE m, or APPEND m


@dataclass
class FileAnalysis:
    file: str
    duration_s: float
    total_chunks: int
    partials: list[PartialUpdate]
    final_text: str

    @property
    def corrections(self) -> int:
        return sum(1 for p in self.partials if p.is_correction)

    @property
    def max_rollback(self) -> int:
        return max((p.removed_chars for p in self.partials), default=0)

    @property
    def avg_rollback(self) -> float:
        rbs = [p.removed_chars for p in self.partials if p.removed_chars > 0]
        return sum(rbs) / len(rbs) if rbs else 0

    @property
    def max_added(self) -> int:
        return max((p.added_chars for p in self.partials), default=0)

    @property
    def total_rollback_chars(self) -> int:
        return sum(p.removed_chars for p in self.partials)

    @property
    def large_replays(self) -> list[PartialUpdate]:
        """Updates where >20 chars were replayed (snap+reanimate)."""
        return [p for p in self.partials if p.removed_chars > 20]


def stream_and_log(path: str) -> FileAnalysis:
    pcm = load_audio_as_pcm(path)
    duration_s = len(pcm) / SAMPLE_RATE
    fname = os.path.basename(path)

    # Create session
    resp = requests.post(
        f"{SIDECAR_URL}/v1/audio/transcriptions/stream",
        json={},
        headers={"Content-Type": "application/json"},
    )
    if not resp.ok:
        raise RuntimeError(f"Failed to create session: {resp.status_code} {resp.text}")
    sid = resp.json()["session_id"]

    partials: list[PartialUpdate] = []
    prev_text = ""
    chunk_idx = 0

    # Feed chunks
    for i in range(0, len(pcm), CHUNK_SAMPLES):
        chunk = pcm[i:i + CHUNK_SAMPLES]
        t0 = time.time()
        resp = requests.post(
            f"{SIDECAR_URL}/v1/audio/transcriptions/stream/{sid}",
            data=chunk.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        elapsed_ms = (time.time() - t0) * 1000

        if not resp.ok:
            chunk_idx += 1
            continue

        text = resp.json().get("text", "")

        # Compute diff against previous
        cp = common_prefix_len(prev_text, text)
        removed_chars = len(prev_text) - cp
        removed_text = prev_text[cp:] if removed_chars > 0 else ""
        added_chars = len(text) - cp
        added_text = text[cp:] if added_chars > 0 else ""
        is_correction = removed_chars > 0

        # Simulate TypewriterAnimator decision
        if is_correction:
            # Snap corrected portion, animate only truly new chars beyond old length
            snap_chars = removed_chars
            animate_chars = max(0, len(text) - len(prev_text))
            if animate_chars > 0:
                tw_action = f"SNAP {snap_chars} + ANIMATE {animate_chars}"
            else:
                tw_action = f"SNAP {snap_chars}"
        else:
            tw_action = f"ANIMATE {added_chars}"

        update = PartialUpdate(
            chunk_idx=chunk_idx,
            elapsed_ms=round(elapsed_ms, 1),
            text=text,
            common_prefix=cp,
            removed_chars=removed_chars,
            removed_text=removed_text,
            added_chars=added_chars,
            added_text=added_text,
            is_correction=is_correction,
            typewriter_action=tw_action,
        )
        partials.append(update)
        prev_text = text
        chunk_idx += 1

    # Stop session
    resp = requests.delete(f"{SIDECAR_URL}/v1/audio/transcriptions/stream/{sid}")
    final_text = resp.json().get("text", prev_text) if resp.ok else prev_text

    return FileAnalysis(
        file=fname,
        duration_s=duration_s,
        total_chunks=chunk_idx,
        partials=partials,
        final_text=final_text,
    )


def print_analysis(analysis: FileAnalysis) -> None:
    print(f"\n{'='*70}")
    print(f"  {analysis.file} ({analysis.duration_s:.1f}s, {analysis.total_chunks} chunks)")
    print(f"{'='*70}")

    for p in analysis.partials:
        tag = " *** CORRECTION" if p.is_correction else ""
        print(f"\n  chunk {p.chunk_idx} ({p.elapsed_ms:.0f}ms){tag}")
        print(f"    text: \"{p.text[:100]}{'...' if len(p.text) > 100 else ''}\"")

        if p.removed_chars > 0:
            print(f"    REMOVED {p.removed_chars}: \"{p.removed_text[:60]}\"")
        if p.added_chars > 0:
            print(f"    ADDED   {p.added_chars}: \"{p.added_text[:60]}\"")

        print(f"    prefix={p.common_prefix} | {p.typewriter_action}")

    # Summary
    print(f"\n  --- Summary ---")
    print(f"  Chunks with corrections: {analysis.corrections}/{len(analysis.partials)}")
    print(f"  Max rollback: {analysis.max_rollback} chars")
    print(f"  Avg rollback (when >0): {analysis.avg_rollback:.0f} chars")
    print(f"  Max chars added in one update: {analysis.max_added}")
    print(f"  Total chars rolled back: {analysis.total_rollback_chars}")

    large = analysis.large_replays
    if large:
        print(f"  Large replays (>20 chars removed): {len(large)}")
        for p in large:
            print(f"    chunk {p.chunk_idx}: removed {p.removed_chars} \"{p.removed_text[:40]}...\"")

    # Final text
    print(f"\n  Final ({len(analysis.final_text)} chars): {analysis.final_text[:120]}{'...' if len(analysis.final_text) > 120 else ''}")


def find_audio_files(args: argparse.Namespace) -> list[str]:
    if args.files:
        return [os.path.expanduser(f) for f in args.files if os.path.isfile(os.path.expanduser(f))]

    candidates: list[str] = []
    yuwp_dir = os.path.expanduser("~/Library/Application Support/Yuwp/recordings/")
    recordings_dir = os.path.expanduser("~/Library/Application Support/Yuwp/recordings/")
    candidates.extend(sorted(glob.glob(os.path.join(yuwp_dir, "*.wav"))))
    candidates.extend(sorted(glob.glob(os.path.join(recordings_dir, "**/*.flac"), recursive=True)))

    n = args.last or 5
    return candidates[-n:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Log streaming partials for rollback analysis")
    parser.add_argument("--files", nargs="+", help="Audio files to analyze")
    parser.add_argument("--last", type=int, default=None, help="Analyze last N audio files")
    parser.add_argument("--port", type=int, default=9748, help="Sidecar port")
    parser.add_argument("--json", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    global SIDECAR_URL
    SIDECAR_URL = f"http://127.0.0.1:{args.port}"

    # Check sidecar is running
    try:
        resp = requests.get(f"{SIDECAR_URL}/v1/info", timeout=3)
        if resp.ok:
            info = resp.json()
            print(f"Sidecar: {info.get('model', 'unknown')}")
    except requests.ConnectionError:
        print(f"ERROR: sidecar not running on port {args.port}")
        print(f"Start it: uv run Sources/sidecar/transcribe.py --serve-only")
        sys.exit(1)

    files = find_audio_files(args)
    if not files:
        print("No audio files found")
        sys.exit(1)

    print(f"Analyzing {len(files)} file(s) for streaming rollback volatility\n")

    all_analyses: list[FileAnalysis] = []
    for path in files:
        analysis = stream_and_log(path)
        print_analysis(analysis)
        all_analyses.append(analysis)

    # Global summary
    print(f"\n{'='*70}")
    print("GLOBAL SUMMARY")
    print(f"{'='*70}")

    total_chunks = sum(a.total_chunks for a in all_analyses)
    total_corrections = sum(a.corrections for a in all_analyses)
    total_partials = sum(len(a.partials) for a in all_analyses)
    all_rollbacks = [p.removed_chars for a in all_analyses for p in a.partials if p.removed_chars > 0]
    all_added = [p.added_chars for a in all_analyses for p in a.partials if p.added_chars > 0]
    all_large = [p for a in all_analyses for p in a.large_replays]

    print(f"  Files: {len(all_analyses)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total partials with text: {total_partials}")
    print(f"  Corrections (rollbacks): {total_corrections}/{total_partials} ({100*total_corrections/max(total_partials,1):.0f}%)")

    if all_rollbacks:
        print(f"  Rollback stats:")
        print(f"    Mean: {sum(all_rollbacks)/len(all_rollbacks):.1f} chars")
        sorted_rb = sorted(all_rollbacks)
        print(f"    Median: {sorted_rb[len(sorted_rb)//2]} chars")
        print(f"    Max: {max(all_rollbacks)} chars")
        print(f"    P95: {sorted_rb[int(len(sorted_rb)*0.95)]} chars")

    if all_added:
        print(f"  Added-per-update stats:")
        print(f"    Mean: {sum(all_added)/len(all_added):.1f} chars")
        sorted_add = sorted(all_added)
        print(f"    Median: {sorted_add[len(sorted_add)//2]} chars")
        print(f"    Max: {max(all_added)} chars")

    if all_large:
        print(f"  Large replays (>20 chars): {len(all_large)}")

    # Volatility score: how jerky does the UI feel?
    # Higher = worse. Weighted by correction frequency and rollback size.
    if total_partials > 0:
        correction_rate = total_corrections / total_partials
        avg_rb = sum(all_rollbacks) / max(len(all_rollbacks), 1)
        volatility = correction_rate * avg_rb
        print(f"\n  Volatility score: {volatility:.1f} (correction_rate={correction_rate:.2f} x avg_rollback={avg_rb:.0f})")
        if volatility < 5:
            print(f"  Assessment: LOW — text is mostly stable")
        elif volatility < 15:
            print(f"  Assessment: MODERATE — occasional corrections visible")
        else:
            print(f"  Assessment: HIGH — frequent large corrections, jerky UX")

    # Save JSON
    if args.json:
        out = []
        for a in all_analyses:
            entry = {
                "file": a.file,
                "duration_s": a.duration_s,
                "total_chunks": a.total_chunks,
                "corrections": a.corrections,
                "max_rollback": a.max_rollback,
                "avg_rollback": round(a.avg_rollback, 1),
                "total_rollback_chars": a.total_rollback_chars,
                "final_text": a.final_text,
                "partials": [
                    {
                        "chunk": p.chunk_idx,
                        "ms": p.elapsed_ms,
                        "text": p.text,
                        "prefix": p.common_prefix,
                        "removed": p.removed_chars,
                        "removed_text": p.removed_text,
                        "added": p.added_chars,
                        "added_text": p.added_text,
                        "correction": p.is_correction,
                        "typewriter": p.typewriter_action,
                    }
                    for p in a.partials
                ],
            }
            out.append(entry)
        with open(args.json, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved to {args.json}")


if __name__ == "__main__":
    main()
