#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = REPO_ROOT / ".build/arm64-apple-macosx/release/asr-stream-test"
YUWP_RECORDINGS_DIR = Path.home() / "Library/Application Support/Yuwp/recordings"
OPPI_DICTATION_DIR = Path.home() / ".config/oppi/dictation"
OUTPUT_DIR = Path("/tmp/yuwp-autoresearch/streaming-quality")

SMOKE_CANARY_CASES: list[tuple[str, str]] = [
    # Tiny suite for ultra-fast smoke checks.
    ("yuwp", "yuwp-2026-04-11_00-13-00.wav"),
    ("oppi", "2026/04/10/dict_5e37cf88a5d6a5ae.flac"),
]

FAST_CANARY_CASES: list[tuple[str, str]] = [
    # Worst mixed-case offenders for fast iteration.
    ("yuwp", "yuwp-2026-04-11_00-13-00.wav"),
    ("oppi", "2026/04/10/dict_5e37cf88a5d6a5ae.flac"),
    ("oppi", "2026/04/06/dict_3466fa362a4bef2f.flac"),
    ("oppi", "2026/04/08/dict_8204dfcc076241fb.flac"),
    ("oppi", "2026/04/08/dict_df27f1db72b17a60.flac"),
]

FULL_CANARY_CASES: list[tuple[str, str]] = [
    ("yuwp", "yuwp-2026-04-07_14-51-05.wav"),
    ("yuwp", "yuwp-2026-04-07_14-52-30.wav"),
    ("yuwp", "yuwp-2026-04-07_14-53-53.wav"),
    ("yuwp", "yuwp-2026-04-08_09-04-36.wav"),
    ("yuwp", "yuwp-2026-04-11_00-08-42.wav"),
    ("yuwp", "yuwp-2026-04-11_00-13-00.wav"),
    ("yuwp", "yuwp-2026-04-11_00-18-24.wav"),
    ("yuwp", "yuwp-2026-04-11_00-09-29.wav"),
    ("yuwp", "yuwp-2026-04-11_00-04-17.wav"),
    ("yuwp", "yuwp-2026-04-11_00-12-02.wav"),
    # Oppi dictation offenders with the worst no-growth behavior.
    ("oppi", "2026/04/10/dict_5e37cf88a5d6a5ae.flac"),
    ("oppi", "2026/04/06/dict_3466fa362a4bef2f.flac"),
    ("oppi", "2026/04/08/dict_8204dfcc076241fb.flac"),
    ("oppi", "2026/04/08/dict_df27f1db72b17a60.flac"),
    ("oppi", "2026/04/05/dict_7e537098e40db67f.flac"),
]

# User preference: genuine pauses are fine. What hurts is when speech resumes or
# keeps going, but old words only appear after more talking. So prioritize
# speech→first-text delay and *recovered* stall time (stalls that later resolve
# into new text), not every quiet plateau.
RECOVERED_NO_GROWTH_SECONDS_WEIGHT = 0.06
RECOVERED_NO_GROWTH_CASE_RATE_WEIGHT = 0.05
MAX_RECOVERED_NO_GROWTH_SECONDS_WEIGHT = 0.035
SPEECH_TO_FIRST_TEXT_WEIGHT = 0.03
FIRST_TEXT_WEIGHT = 0.004
FINALIZATION_WEIGHT = 0.001


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming-quality canary benchmark")
    parser.add_argument("--bin", default=str(DEFAULT_BIN), help="Path to asr-stream-test binary")
    parser.add_argument(
        "--out-dir",
        default=str(OUTPUT_DIR),
        help="Directory for per-file JSON artifacts",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Write compact benchmark summary JSON instead of pretty",
    )
    parser.add_argument(
        "--suite",
        choices=["smoke", "fast", "full"],
        default="fast",
        help="Benchmark suite to run (default: fast)",
    )
    parser.add_argument(
        "--full-session-retranscribe",
        action="store_true",
        help="Opt into full-session batch finalization on stop",
    )
    return parser.parse_args()


def metric(name: str, value: float | int) -> None:
    print(f"METRIC {name}={value}")


def case_root(source: str) -> Path:
    if source == "yuwp":
        return YUWP_RECORDINGS_DIR
    if source == "oppi":
        return OPPI_DICTATION_DIR
    raise ValueError(f"unknown case source: {source}")


def resolve_case(case: tuple[str, str]) -> Path:
    source, rel_path = case
    return case_root(source) / rel_path


def materialize_audio(input_path: Path, cache_dir: Path) -> Path:
    if input_path.suffix.lower() == ".wav":
        return input_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / f"{input_path.stem}.wav"
    if wav_path.exists() and wav_path.stat().st_mtime >= input_path.stat().st_mtime:
        return wav_path

    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(input_path), str(wav_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {input_path}:\n{proc.stderr[-2000:]}"
        )
    return wav_path


def run_timed(cmd: list[str]) -> tuple[subprocess.CompletedProcess[str], dict[str, float | int | None]]:
    wrapped = ["/usr/bin/time", "-l"] + cmd
    proc = subprocess.run(wrapped, capture_output=True, text=True)
    stderr_text = proc.stderr

    user_s = None
    sys_s = None
    max_rss_bytes = None
    for line in stderr_text.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[1] == "real" and parts[3] == "user" and parts[5] == "sys":
            try:
                user_s = float(parts[2])
                sys_s = float(parts[4])
            except Exception:
                pass
        elif "maximum resident set size" in line:
            try:
                max_rss_bytes = int(parts[0])
            except Exception:
                pass

    if proc.returncode != 0:
        return proc, {
            "user_cpu_s": user_s,
            "sys_cpu_s": sys_s,
            "max_rss_bytes": max_rss_bytes,
        }

    cleaned_stderr_lines = []
    for line in stderr_text.splitlines():
        parts = line.split()
        if (len(parts) >= 6 and parts[1] == "real" and parts[3] == "user" and parts[5] == "sys") or "maximum resident set size" in line:
            continue
        cleaned_stderr_lines.append(line)
    proc = subprocess.CompletedProcess(proc.args, proc.returncode, proc.stdout, "\n".join(cleaned_stderr_lines))
    return proc, {
        "user_cpu_s": user_s,
        "sys_cpu_s": sys_s,
        "max_rss_bytes": max_rss_bytes,
    }


def main() -> int:
    args = parse_args()
    bin_path = Path(args.bin)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bin_path.exists():
        print(f"benchmark binary not found: {bin_path}", file=sys.stderr)
        return 2

    if args.suite == "smoke":
        selected_cases = SMOKE_CANARY_CASES
    elif args.suite == "fast":
        selected_cases = FAST_CANARY_CASES
    else:
        selected_cases = FULL_CANARY_CASES
    files = [resolve_case(case) for case in selected_cases]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        print("missing canary files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    started_at = time.time()
    reports: list[dict] = []
    timings: list[dict[str, float | int | None]] = []
    cache_dir = out_dir / "audio-cache"

    for idx, (case, input_path) in enumerate(zip(selected_cases, files), 1):
        source, rel_path = case
        audio_path = materialize_audio(input_path, cache_dir)
        json_path = out_dir / f"{source}-{input_path.stem}.json"
        cmd = [str(bin_path), str(audio_path), "--compact", "--json-output", str(json_path)]
        if args.full_session_retranscribe:
            cmd.append("--full-session-retranscribe")
        print(f"[bench] ({idx}/{len(files)}) {source}:{rel_path}", file=sys.stderr)
        proc, timing = run_timed(cmd)
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
            return proc.returncode
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            print(f"failed to decode JSON for {source}:{rel_path}: {exc}", file=sys.stderr)
            print(proc.stdout[:1000], file=sys.stderr)
            return 1
        report["_file"] = input_path.name
        report["_source"] = source
        report["_case_path"] = rel_path
        report["_process"] = timing
        reports.append(report)
        timings.append(timing)

    wers = [r["accuracy"]["wordErrorRate"] for r in reports]
    first_text = [r["streaming"]["firstTextAudioSec"] for r in reports if r["streaming"].get("firstTextAudioSec") is not None]
    no_growth_chunks = [r["streaming"]["speechChunksWithoutGrowth"] for r in reports]
    no_growth_seconds = [r["streaming"]["speechSecondsWithoutGrowth"] for r in reports]
    recovered_no_growth_chunks = [r["streaming"].get("recoveredSpeechNoGrowthChunks", 0) for r in reports]
    recovered_no_growth_seconds = [r["streaming"].get("recoveredSpeechNoGrowthSec", 0) for r in reports]
    final_added = [r["streaming"]["finalizationAddedWords"] for r in reports]
    finalization_seconds = [r["streaming"].get("finalizationSec", 0.0) for r in reports]
    normalized_exact_rate = sum(1 for r in reports if r["accuracy"]["normalizedExactMatch"]) / len(reports)
    mean_wer = statistics.mean(wers)
    p90_wer = percentile(wers, 0.90)
    speech_to_first_text = [
        r["streaming"]["speechToFirstTextSec"]
        for r in reports
        if r["streaming"].get("speechToFirstTextSec") is not None
    ]
    mean_first_text = statistics.mean(first_text) if first_text else 99.0
    mean_speech_to_first_text = (
        statistics.mean(speech_to_first_text) if speech_to_first_text else mean_first_text
    )
    mean_no_growth_chunks = statistics.mean(no_growth_chunks)
    mean_no_growth_seconds = statistics.mean(no_growth_seconds)
    mean_recovered_no_growth_chunks = statistics.mean(recovered_no_growth_chunks)
    mean_recovered_no_growth_seconds = statistics.mean(recovered_no_growth_seconds)
    max_no_growth_run = max(r["streaming"]["maxConsecutiveSpeechChunksWithoutGrowth"] for r in reports)
    max_no_growth_seconds = max(r["streaming"]["maxConsecutiveSpeechNoGrowthSec"] for r in reports)
    max_recovered_no_growth_seconds = max(r["streaming"].get("maxRecoveredSpeechNoGrowthSec", 0) for r in reports)
    recordings_with_any_no_growth = sum(1 for r in reports if r["streaming"]["speechChunksWithoutGrowth"] > 0)
    no_growth_case_rate = recordings_with_any_no_growth / len(reports)
    recordings_with_recovered_stall = sum(1 for r in reports if r["streaming"].get("recoveredSpeechNoGrowthChunks", 0) > 0)
    recovered_no_growth_case_rate = recordings_with_recovered_stall / len(reports)
    mean_final_added = statistics.mean(final_added)

    stream_quality_score = (
        mean_wer
        + RECOVERED_NO_GROWTH_SECONDS_WEIGHT * mean_recovered_no_growth_seconds
        + RECOVERED_NO_GROWTH_CASE_RATE_WEIGHT * recovered_no_growth_case_rate
        + MAX_RECOVERED_NO_GROWTH_SECONDS_WEIGHT * max_recovered_no_growth_seconds
        + SPEECH_TO_FIRST_TEXT_WEIGHT * mean_speech_to_first_text
        + FIRST_TEXT_WEIGHT * mean_first_text
        + FINALIZATION_WEIGHT * mean_final_added
    )

    worst = sorted(
        reports,
        key=lambda r: (
            r["accuracy"]["wordErrorRate"],
            r["streaming"].get("recoveredSpeechNoGrowthSec") or 0,
            r["streaming"].get("speechToFirstTextSec") or -1,
            r["streaming"].get("maxRecoveredSpeechNoGrowthSec") or 0,
            r["streaming"]["finalizationAddedWords"],
        ),
        reverse=True,
    )[:5]

    user_cpu = [t["user_cpu_s"] for t in timings if t.get("user_cpu_s") is not None]
    sys_cpu = [t["sys_cpu_s"] for t in timings if t.get("sys_cpu_s") is not None]
    rss_bytes = [t["max_rss_bytes"] for t in timings if t.get("max_rss_bytes") is not None]

    summary = {
        "generated_at_epoch": time.time(),
        "elapsed_wall_sec": time.time() - started_at,
        "binary": str(bin_path),
        "input_roots": {
            "yuwp": str(YUWP_RECORDINGS_DIR),
            "oppi": str(OPPI_DICTATION_DIR),
        },
        "score_weights": {
            "mean_wer": 1.0,
            "mean_recovered_speech_no_growth_seconds": RECOVERED_NO_GROWTH_SECONDS_WEIGHT,
            "recovered_no_growth_case_rate": RECOVERED_NO_GROWTH_CASE_RATE_WEIGHT,
            "max_recovered_no_growth_seconds": MAX_RECOVERED_NO_GROWTH_SECONDS_WEIGHT,
            "mean_speech_to_first_text_s": SPEECH_TO_FIRST_TEXT_WEIGHT,
            "mean_first_text_s": FIRST_TEXT_WEIGHT,
            "mean_finalization_added_words": FINALIZATION_WEIGHT,
        },
        "execution": {
            "assumed_device": "gpu-default-mlx",
            "mean_user_cpu_s": statistics.mean(user_cpu) if user_cpu else None,
            "mean_sys_cpu_s": statistics.mean(sys_cpu) if sys_cpu else None,
            "total_user_cpu_s": sum(user_cpu) if user_cpu else None,
            "total_sys_cpu_s": sum(sys_cpu) if sys_cpu else None,
            "max_rss_mb": (max(rss_bytes) / 1_048_576) if rss_bytes else None,
            "mean_rss_mb": (statistics.mean(rss_bytes) / 1_048_576) if rss_bytes else None,
        },
        "suite": args.suite,
        "full_session_retranscribe": args.full_session_retranscribe,
        "cases": [
            {"source": source, "path": rel_path}
            for source, rel_path in selected_cases
        ],
        "stream_quality_score": stream_quality_score,
        "mean_wer": mean_wer,
        "p90_wer": p90_wer,
        "normalized_exact_rate": normalized_exact_rate,
        "mean_first_text_s": mean_first_text,
        "mean_speech_to_first_text_s": mean_speech_to_first_text,
        "mean_speech_no_growth": mean_no_growth_chunks,
        "mean_speech_no_growth_seconds": mean_no_growth_seconds,
        "mean_recovered_speech_no_growth_chunks": mean_recovered_no_growth_chunks,
        "mean_recovered_speech_no_growth_seconds": mean_recovered_no_growth_seconds,
        "no_growth_case_rate": no_growth_case_rate,
        "recordings_with_any_no_growth": recordings_with_any_no_growth,
        "recovered_no_growth_case_rate": recovered_no_growth_case_rate,
        "recordings_with_recovered_stall": recordings_with_recovered_stall,
        "max_no_growth_run": max_no_growth_run,
        "max_no_growth_seconds": max_no_growth_seconds,
        "max_recovered_no_growth_seconds": max_recovered_no_growth_seconds,
        "mean_finalization_added_words": mean_final_added,
        "mean_segment_commits": statistics.mean(r["streaming"]["segmentCommitCount"] for r in reports),
        "mean_finalization_s": statistics.mean(finalization_seconds),
        "p90_finalization_s": percentile(finalization_seconds, 0.90),
        "worst_cases": [
            {
                "file": r["_file"],
                "source": r["_source"],
                "case_path": r["_case_path"],
                "wer": r["accuracy"]["wordErrorRate"],
                "speech_no_growth": r["streaming"]["speechChunksWithoutGrowth"],
                "speech_no_growth_seconds": r["streaming"].get("speechSecondsWithoutGrowth"),
                "recovered_speech_no_growth_seconds": r["streaming"].get("recoveredSpeechNoGrowthSec"),
                "speech_to_first_text_s": r["streaming"].get("speechToFirstTextSec"),
                "first_text_s": r["streaming"].get("firstTextAudioSec"),
                "finalization_added_words": r["streaming"]["finalizationAddedWords"],
            }
            for r in worst
        ],
        "reports": reports,
    }
    summary_path = out_dir / "canary-summary.json"
    summary_path.write_text(json.dumps(summary, indent=None if args.compact_json else 2, sort_keys=not args.compact_json))

    metric("stream_quality_score", f"{stream_quality_score:.6f}")
    metric("mean_wer", f"{mean_wer:.6f}")
    metric("p90_wer", f"{p90_wer:.6f}")
    metric("normalized_exact_rate", f"{normalized_exact_rate:.6f}")
    metric("mean_first_text_s", f"{mean_first_text:.6f}")
    metric("mean_speech_to_first_text_s", f"{mean_speech_to_first_text:.6f}")
    metric("mean_speech_no_growth", f"{mean_no_growth_chunks:.6f}")
    metric("mean_speech_no_growth_seconds", f"{mean_no_growth_seconds:.6f}")
    metric("mean_recovered_speech_no_growth_seconds", f"{mean_recovered_no_growth_seconds:.6f}")
    metric("max_no_growth_run", max_no_growth_run)
    metric("max_no_growth_seconds", f"{max_no_growth_seconds:.6f}")
    metric("max_recovered_no_growth_seconds", f"{max_recovered_no_growth_seconds:.6f}")
    metric("mean_finalization_added_words", f"{mean_final_added:.6f}")
    metric("mean_finalization_s", f"{summary['mean_finalization_s']:.6f}")
    metric("p90_finalization_s", f"{summary['p90_finalization_s']:.6f}")
    metric("mean_segment_commits", f"{summary['mean_segment_commits']:.6f}")
    if summary["execution"]["mean_user_cpu_s"] is not None:
        metric("mean_user_cpu_s", f"{summary['execution']['mean_user_cpu_s']:.6f}")
    if summary["execution"]["mean_sys_cpu_s"] is not None:
        metric("mean_sys_cpu_s", f"{summary['execution']['mean_sys_cpu_s']:.6f}")
    if summary["execution"]["max_rss_mb"] is not None:
        metric("max_rss_mb", f"{summary['execution']['max_rss_mb']:.2f}")
    metric("canary_case_count", len(reports))
    metric("bench_elapsed_s", f"{summary['elapsed_wall_sec']:.6f}")
    print(f"[bench] summary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
