#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["mlx-lm", "numpy"]
# ///
"""
Benchmark LLM cleanup vs ASR batch correction on streaming transcripts.

Compares three finalization strategies for 0.6B-4bit streaming output:
  1. Raw streaming text (no correction)
  2. 1.7B ASR batch retranscription (current approach)
  3. Small LLM cleanup (Qwen3 0.6B/1.7B text model)

Uses saved benchmark results from results/bench-*.json.

Usage:
  # Default: Qwen3-0.6B-4bit LLM, last benchmark results
  uv run scripts/bench-llm-cleanup.py

  # Try larger LLM
  uv run scripts/bench-llm-cleanup.py --llm mlx-community/Qwen3-1.7B-4bit

  # Custom benchmark file
  uv run scripts/bench-llm-cleanup.py --bench results/bench-20260407-full.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


CLEANUP_PROMPT = """Clean up this voice dictation transcript. Fix:
- Remove filler words (um, uh, like, you know) unless they add meaning
- Fix repeated/stuttered words ("I think, I think" → "I think")
- Fix punctuation and capitalization
- Preserve the original meaning, language, and code-switching exactly
- Do NOT paraphrase, summarize, or add words that weren't said
- Output ONLY the cleaned text, nothing else

Transcript:
{text}"""

CLEANUP_PROMPT_BILINGUAL = """Clean up this voice dictation transcript. The speaker switches between English and Chinese. Fix:
- Remove filler words in both languages
- Fix repeated/stuttered words
- Fix punctuation and capitalization
- Keep English as English and Chinese as Chinese — do not translate
- Preserve code-switching boundaries exactly as spoken
- Output ONLY the cleaned text, nothing else

Transcript:
{text}"""


def count_cjk(text: str) -> int:
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')


def has_cjk(text: str) -> bool:
    return count_cjk(text) > 0


def load_llm(model_name: str):
    """Load LLM via mlx-lm."""
    from mlx_lm import load
    print(f"Loading LLM: {model_name}...")
    t0 = time.time()
    model, tokenizer = load(model_name)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def llm_cleanup(model, tokenizer, text: str) -> tuple[str, float]:
    """Run LLM cleanup on transcript text. Returns (cleaned_text, elapsed_ms)."""
    from mlx_lm import generate

    prompt_template = CLEANUP_PROMPT_BILINGUAL if has_cjk(text) else CLEANUP_PROMPT
    prompt = prompt_template.format(text=text)

    # Build chat messages
    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizers may not support enable_thinking
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    t0 = time.time()
    result = generate(
        model, tokenizer,
        prompt=chat_prompt,
        max_tokens=len(text) * 2,  # generous limit
        verbose=False,
    )
    elapsed_ms = (time.time() - t0) * 1000

    # Strip thinking tags if present (Qwen3 sometimes wraps in <think>)
    cleaned = result.strip()
    if "<think>" in cleaned:
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()

    return cleaned, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLM cleanup vs ASR batch")
    parser.add_argument("--llm", default="mlx-community/Qwen3-0.6B-4bit",
                        help="LLM model for cleanup (default: Qwen3-0.6B-4bit)")
    parser.add_argument("--llm2", default=None,
                        help="Second LLM to compare (optional)")
    parser.add_argument("--bench", default=None,
                        help="Path to benchmark JSON (default: latest in results/)")
    args = parser.parse_args()

    # Find benchmark results
    if args.bench:
        bench_path = args.bench
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        jsons = sorted(results_dir.glob("bench-*.json"))
        if not jsons:
            print("No benchmark results found in results/")
            sys.exit(1)
        bench_path = str(jsons[-1])

    print(f"Loading benchmark: {bench_path}")
    with open(bench_path) as f:
        bench_data = json.load(f)

    # Find 0.6B-4bit and 1.7B-bf16 results
    stream_model = None
    batch_model = None
    for entry in bench_data:
        if "0.6B-4bit" in entry["model"]:
            stream_model = entry
        elif "1.7B-bf16" in entry["model"]:
            batch_model = entry

    if not stream_model:
        print("No 0.6B-4bit results in benchmark data")
        sys.exit(1)

    print(f"Streaming model: {stream_model['model']}")
    if batch_model:
        print(f"Batch model: {batch_model['model']}")
    print(f"Files: {len(stream_model['files'])}")

    # Load LLM(s)
    llm_models = []
    model1, tok1 = load_llm(args.llm)
    llm_models.append((args.llm, model1, tok1))

    if args.llm2:
        model2, tok2 = load_llm(args.llm2)
        llm_models.append((args.llm2, model2, tok2))

    # Process each file
    print(f"\n{'='*70}")
    print("RUNNING LLM CLEANUP")
    print(f"{'='*70}")

    results = []
    total_llm_ms = {name: 0.0 for name, _, _ in llm_models}

    for i, file_entry in enumerate(stream_model["files"]):
        fname = file_entry["file"]
        dur = file_entry["duration_s"]
        stream_text = file_entry.get("stream_text", file_entry.get("text", ""))
        batch_text = file_entry.get("batch_text", "")
        saved_text = file_entry.get("saved_text", "")

        if not stream_text:
            continue

        print(f"\n  [{i+1}/{len(stream_model['files'])}] {fname} ({dur:.1f}s)")
        print(f"    stream ({len(stream_text)} chars): {stream_text[:80]}...")

        entry = {
            "file": fname,
            "duration_s": dur,
            "stream_text": stream_text,
            "batch_text": batch_text,
            "saved_text": saved_text,
            "llm_results": {},
        }

        for llm_name, model, tokenizer in llm_models:
            short = llm_name.split("/")[-1]
            cleaned, elapsed_ms = llm_cleanup(model, tokenizer, stream_text)
            total_llm_ms[llm_name] += elapsed_ms
            entry["llm_results"][short] = {
                "text": cleaned,
                "elapsed_ms": round(elapsed_ms, 1),
            }
            print(f"    {short} ({elapsed_ms:.0f}ms, {len(cleaned)} chars): {cleaned[:80]}...")

        if batch_text and batch_text.strip() != stream_text.strip():
            print(f"    1.7B batch ({len(batch_text)} chars): {batch_text[:80]}...")
        if saved_text:
            print(f"    saved ({len(saved_text)} chars): {saved_text[:80]}...")

        results.append(entry)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total_audio = sum(r["duration_s"] for r in results)
    print(f"  Files: {len(results)}")
    print(f"  Total audio: {total_audio:.0f}s")

    for llm_name, _, _ in llm_models:
        short = llm_name.split("/")[-1]
        total_ms = total_llm_ms[llm_name]
        avg_ms = total_ms / len(results) if results else 0
        print(f"\n  {short}:")
        print(f"    Total cleanup time: {total_ms/1000:.1f}s")
        print(f"    Avg per file: {avg_ms:.0f}ms")

    # Compare: how often does LLM match batch better than raw streaming?
    if batch_model:
        print(f"\n  Quality comparison (vs saved transcript when available):")

        for llm_name, _, _ in llm_models:
            short = llm_name.split("/")[-1]
            llm_closer = 0
            batch_closer = 0
            same = 0

            for r in results:
                saved = r["saved_text"]
                if not saved:
                    continue

                stream = r["stream_text"]
                batch = r["batch_text"]
                llm_text = r["llm_results"].get(short, {}).get("text", "")

                if not llm_text:
                    continue

                # Simple character-level distance
                def char_distance(a: str, b: str) -> int:
                    # Levenshtein-ish: just use length diff + first-divergence position
                    min_len = min(len(a), len(b))
                    diffs = sum(1 for i in range(min_len) if a[i] != b[i])
                    return diffs + abs(len(a) - len(b))

                d_stream = char_distance(stream.strip(), saved.strip())
                d_batch = char_distance(batch.strip(), saved.strip())
                d_llm = char_distance(llm_text.strip(), saved.strip())

                if d_llm < d_batch:
                    llm_closer += 1
                elif d_batch < d_llm:
                    batch_closer += 1
                else:
                    same += 1

            total = llm_closer + batch_closer + same
            if total > 0:
                print(f"    {short}: closer to saved {llm_closer}/{total}, "
                      f"batch closer {batch_closer}/{total}, tie {same}/{total}")

    # Save results
    out_path = Path(__file__).resolve().parent.parent / "results" / "llm-cleanup-bench.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
