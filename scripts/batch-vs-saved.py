#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["mlx-audio"]
# ///
"""
Compare saved streaming transcripts against batch retranscription.

Each dictation JSON already has the streaming transcript. This script
batch-retranscribes every FLAC and compares. No sidecar needed.

Usage:
  uv run scripts/batch-vs-saved.py              # all files
  uv run scripts/batch-vs-saved.py --last 10    # last 10
  uv run scripts/batch-vs-saved.py --json       # machine-readable
"""

import argparse
import glob
import json
import os
import sys
import time


MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DICTATION_DIR = os.path.expanduser("~/Library/Application Support/Yuwp/recordings/")

_model = None

def get_model():
    global _model
    if _model is None:
        from mlx_audio.stt.generate import load_model
        _model = load_model(MODEL)
    return _model


def batch_transcribe(path: str) -> tuple[str, float]:
    from mlx_audio.stt.generate import generate_transcription
    model = get_model()
    t0 = time.time()
    result = generate_transcription(model, path)
    return result.text, time.time() - t0


def count_cjk(text: str) -> int:
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')


def find_pairs() -> list[tuple[str, dict]]:
    """Find all (flac_path, metadata) pairs."""
    pairs = []
    for jf in sorted(glob.glob(os.path.join(DICTATION_DIR, "**/*.json"), recursive=True)):
        if "dictionary" in jf or "benchmark" in jf:
            continue
        try:
            meta = json.load(open(jf))
        except Exception:
            continue
        if "transcript" not in meta or "durationMs" not in meta:
            continue
        flac = jf.replace(".json", ".flac")
        if not os.path.exists(flac):
            wav = jf.replace(".json", ".wav")
            if os.path.exists(wav):
                flac = wav
            else:
                continue
        pairs.append((flac, meta))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-duration", type=float, default=0,
                        help="Skip files shorter than N seconds")
    args = parser.parse_args()

    pairs = find_pairs()
    if args.last:
        pairs = pairs[-args.last:]

    if args.min_duration > 0:
        pairs = [(f, m) for f, m in pairs if m["durationMs"] / 1000 >= args.min_duration]

    print(f"Comparing {len(pairs)} dictation files (batch vs saved streaming)")
    print(f"Model: {MODEL}\n")

    results = []
    total_audio = 0
    total_batch_time = 0
    total_cjk_lost = 0
    total_exact = 0
    divergent = []

    for i, (flac, meta) in enumerate(pairs):
        dur_s = meta["durationMs"] / 1000
        stream_text = meta["transcript"]
        name = meta.get("audioId", os.path.basename(flac))

        print(f"[{i+1}/{len(pairs)}] {name} ({dur_s:.1f}s)...", end=" ", flush=True)

        try:
            batch_text, batch_time = batch_transcribe(flac)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        total_audio += dur_s
        total_batch_time += batch_time

        s_cjk = count_cjk(stream_text)
        b_cjk = count_cjk(batch_text)
        cjk_lost = max(0, b_cjk - s_cjk)
        total_cjk_lost += cjk_lost
        exact = stream_text.strip() == batch_text.strip()
        if exact:
            total_exact += 1

        status = "OK" if exact else "DIFF"
        extras = []
        if cjk_lost > 0:
            extras.append(f"CJK lost: {cjk_lost}")
        if abs(len(stream_text) - len(batch_text)) > 20:
            extras.append(f"len: {len(stream_text)}→{len(batch_text)}")

        print(f"{batch_time:.2f}s  {status}" + (f"  ({', '.join(extras)})" if extras else ""))

        entry = {
            "audioId": name,
            "duration_s": dur_s,
            "batch_time_s": round(batch_time, 3),
            "exact_match": exact,
            "stream_len": len(stream_text),
            "batch_len": len(batch_text),
            "stream_cjk": s_cjk,
            "batch_cjk": b_cjk,
            "cjk_lost": cjk_lost,
        }
        results.append(entry)

        if not exact:
            entry["stream_text"] = stream_text
            entry["batch_text"] = batch_text
            divergent.append(entry)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Files: {len(results)}")
    print(f"  Total audio: {total_audio:.0f}s ({total_audio/60:.1f} min)")
    print(f"  Total batch time: {total_batch_time:.1f}s ({total_batch_time/max(total_audio,1):.4f}x realtime)")
    print(f"  Exact matches: {total_exact}/{len(results)} ({100*total_exact/max(len(results),1):.0f}%)")
    print(f"  CJK chars lost (translation): {total_cjk_lost}")
    print(f"  Avg batch-on-stop overhead: {total_batch_time/max(len(results),1):.2f}s per session")

    if divergent:
        print(f"\n  Top divergences (by length delta):")
        divergent.sort(key=lambda x: abs(x["stream_len"] - x["batch_len"]), reverse=True)
        for d in divergent[:10]:
            delta = d["batch_len"] - d["stream_len"]
            cjk_note = f", CJK lost: {d['cjk_lost']}" if d["cjk_lost"] > 0 else ""
            print(f"    {d['audioId']} ({d['duration_s']:.0f}s): {delta:+d} chars{cjk_note}")
            # Show first 100 chars of each for quick comparison
            if d.get("batch_text") and d.get("stream_text"):
                st = d["stream_text"][:80].replace("\n", " ")
                bt = d["batch_text"][:80].replace("\n", " ")
                if st != bt:
                    print(f"      stream: {st}...")
                    print(f"      batch:  {bt}...")

    if args.json:
        out = {
            "summary": {
                "files": len(results),
                "total_audio_s": round(total_audio, 1),
                "total_batch_s": round(total_batch_time, 1),
                "exact_matches": total_exact,
                "cjk_lost": total_cjk_lost,
            },
            "results": results,
        }
        print("\n" + json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
