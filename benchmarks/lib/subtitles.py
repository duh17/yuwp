#!/usr/bin/env -S uv run --python 3.14 --script
"""
Stress-test timed `response_format=json` batch transcription against a set of local files.

Examples:
  uv run benchmarks/cli.py subtitles \
    --manifest benchmarks/fixtures/subtitles-manifest.example.json

  uv run benchmarks/cli.py subtitles \
    --file /tmp/a.m4a --file /tmp/b.m4a \
    --base-url http://127.0.0.1:7936
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:7936"
DEFAULT_OUT_DIR = Path("/tmp/yuwp-subtitle-bench")
DEFAULT_GAP_WARN_SEC = 2.0
DEFAULT_OVERLAP_TOLERANCE_SEC = 0.05


class SampleFailure(Exception):
    pass


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def sanitize_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-")
    return slug or "sample"


def fetch_info(base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/info"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"manifest {path} must contain a non-empty 'samples' list")
    out: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"manifest sample #{index} must be an object")
        out.append(sample)
    return out


def build_samples(files: list[str], manifest: str | None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if manifest:
        samples.extend(load_manifest(Path(manifest)))
    for file in files:
        p = Path(file)
        samples.append({"name": p.stem, "file": str(p)})
    if not samples:
        raise ValueError("provide --manifest or at least one --file")
    return samples


def run_subtitle_request(base_url: str, sample: dict[str, Any]) -> tuple[dict[str, Any], float, str]:
    file_path = Path(str(sample["file"])).expanduser()
    if not file_path.exists():
        raise SampleFailure(f"file not found: {file_path}")

    cmd = [
        "/usr/bin/curl",
        "-sS",
        "-X",
        "POST",
        f"{base_url.rstrip('/')}/v1/audio/transcriptions",
        "-F",
        f"file=@{file_path}",
        "-F",
        "response_format=json",
    ]

    for key in ("text", "language", "max_words_per_line", "max_duration", "pause_threshold"):
        value = sample.get(key)
        if value is not None and value != "":
            cmd.extend(["-F", f"{key}={value}"])

    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - started

    if proc.returncode != 0:
        raise SampleFailure(proc.stderr.strip() or f"curl failed with exit code {proc.returncode}")

    stdout = proc.stdout.strip()
    if not stdout:
        raise SampleFailure("empty response body")

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise SampleFailure(f"invalid JSON response: {exc}") from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise SampleFailure(str(payload["error"]))
    if not isinstance(payload, dict):
        raise SampleFailure(f"expected subtitle JSON object, got {type(payload).__name__}")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise SampleFailure("subtitle JSON object is missing a 'segments' array")

    return payload, wall_seconds, stdout


def analyze_subtitles(items: list[dict[str, Any]], wall_seconds: float, gap_warn_sec: float) -> dict[str, Any]:
    non_monotonic = 0
    empty_text = 0
    invalid_ranges = 0
    overlap_count = 0
    gap_count = 0
    max_gap = 0.0
    max_overlap = 0.0
    word_count = 0
    char_count = 0

    first_start = None
    last_end = None
    prev_end = None

    for item in items:
        text = str(item.get("text") or "").strip()
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or 0.0)

        if not text:
            empty_text += 1
        word_count += len(text.split())
        char_count += len(text)

        if end < start:
            invalid_ranges += 1
        if first_start is None:
            first_start = start
        last_end = end

        if prev_end is not None:
            if start + DEFAULT_OVERLAP_TOLERANCE_SEC < prev_end:
                overlap = prev_end - start
                overlap_count += 1
                max_overlap = max(max_overlap, overlap)
            gap = start - prev_end
            if gap > gap_warn_sec:
                gap_count += 1
                max_gap = max(max_gap, gap)
            if start < prev_end:
                non_monotonic += 1
        prev_end = end

    duration = float(last_end or 0.0)
    realtime_factor = (duration / wall_seconds) if wall_seconds > 0 and duration > 0 else None

    return {
        "entries": len(items),
        "first_start": first_start,
        "last_end": last_end,
        "duration_from_subtitles": duration,
        "wall_seconds": wall_seconds,
        "realtime_factor": realtime_factor,
        "gap_warn_seconds": gap_warn_sec,
        "gap_count": gap_count,
        "max_gap": max_gap,
        "overlap_count": overlap_count,
        "max_overlap": max_overlap,
        "non_monotonic_count": non_monotonic,
        "invalid_range_count": invalid_ranges,
        "empty_text_count": empty_text,
        "word_count": word_count,
        "char_count": char_count,
    }


def summarize_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value):
        return f"{value:.2f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress-test timed batch transcription via /v1/audio/transcriptions")
    parser.add_argument("--manifest", help="JSON manifest with a top-level 'samples' array")
    parser.add_argument("--file", action="append", default=[], help="Local audio/video file to test (repeatable)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Server base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--gap-warn-sec", type=float, default=DEFAULT_GAP_WARN_SEC, help=f"Flag gaps larger than this (default: {DEFAULT_GAP_WARN_SEC})")
    args = parser.parse_args()

    try:
        samples = build_samples(args.file, args.manifest)
    except Exception as exc:
        eprint(f"error: {exc}")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        server_info = fetch_info(args.base_url)
    except urllib.error.URLError as exc:
        eprint(f"error: failed to reach {args.base_url}: {exc}")
        return 2

    eprint(
        "[bench-subtitles] server:",
        json.dumps(
            {
                "status": server_info.get("status"),
                "model": server_info.get("model"),
                "final_accuracy_pass_model": server_info.get("final_accuracy_pass_model"),
                "aligner": server_info.get("aligner"),
                "vad": server_info.get("vad"),
            },
            sort_keys=True,
        ),
    )

    results: list[dict[str, Any]] = []
    failures = 0

    for sample in samples:
        name = str(sample.get("name") or Path(str(sample["file"])).stem)
        slug = sanitize_name(name)
        raw_path = out_dir / f"{slug}.subtitles.json"
        sample_result: dict[str, Any] = {
            "name": name,
            "file": str(Path(str(sample["file"])).expanduser()),
            "source_url": sample.get("source_url"),
        }

        eprint(f"[bench-subtitles] running {name} -> {sample_result['file']}")

        try:
            payload, wall_seconds, raw_json = run_subtitle_request(args.base_url, sample)
            raw_path.write_text(raw_json)
            metrics = analyze_subtitles(payload["segments"], wall_seconds, args.gap_warn_sec)
            metrics["transcript_length"] = len(str(payload.get("text") or ""))
            metrics["language"] = payload.get("language")
            metrics["reported_duration"] = payload.get("duration")
            sample_result.update(metrics)
            sample_result["status"] = "ok"
            sample_result["raw_subtitles_json"] = str(raw_path)
            eprint(
                "[bench-subtitles]",
                name,
                f"ok wall={summarize_metric(metrics['wall_seconds'])}s",
                f"rtf={summarize_metric(metrics['realtime_factor'])}x",
                f"entries={metrics['entries']}",
                f"gaps={metrics['gap_count']}",
                f"overlaps={metrics['overlap_count']}",
            )
        except Exception as exc:
            failures += 1
            sample_result["status"] = "error"
            sample_result["error"] = str(exc)
            eprint(f"[bench-subtitles] {name} FAILED: {exc}")

        results.append(sample_result)

    summary = {
        "base_url": args.base_url,
        "generated_at_unix": time.time(),
        "server_info": server_info,
        "sample_count": len(results),
        "failure_count": failures,
        "results": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))
    eprint(f"[bench-subtitles] summary -> {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
