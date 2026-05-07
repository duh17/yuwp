#!/usr/bin/env -S uv run --python 3.14 --script
"""Unified ASR benchmark front door.

This module intentionally keeps the scenario CLI small and delegates the heavy
benchmark implementations to the existing focused runners while we continue to
shrink their internals.
"""

from __future__ import annotations

import importlib
import sys

SCENARIOS: dict[str, tuple[str, str, str]] = {
    "load": (
        "Concurrent local yuwp-asr serve load/latency benchmark",
        "lib.native_server",
        "benchmarks/cli.py asr load small --balanced 5 --concurrency 1 2 4",
    ),
    "quality": (
        "Streaming quality canary against known fixtures",
        "lib.stream_quality",
        "benchmarks/cli.py asr quality --case jfk",
    ),
    "compare": (
        "Compare yuwp-asr against local tools such as mlx-audio/qwen-asr",
        "lib.batch_compare",
        "benchmarks/cli.py asr compare --audio Tests/fixtures/jfk.wav --yuwp-model small",
    ),
    "remote": (
        "Compare local yuwp-asr against an OpenAI-compatible ASR endpoint",
        "lib.openai_asr_compare",
        "benchmarks/cli.py asr remote --remote-url http://127.0.0.1:8000",
    ),
    "subtitles": (
        "Exercise timed subtitle output from /v1/audio/transcriptions",
        "lib.subtitles",
        "benchmarks/cli.py asr subtitles --file sample.wav",
    ),
}


def print_usage() -> None:
    print("Usage: benchmarks/cli.py asr <scenario> [options]\n")
    print("Scenarios:")
    for name, (summary, _, example) in SCENARIOS.items():
        print(f"  {name:<10} {summary}")
        print(f"             e.g. {example}")
    print("\nRun `benchmarks/cli.py asr <scenario> --help` for scenario-specific options.")


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"--help", "-h", "help"}:
        print_usage()
        return 0

    scenario = argv[0]
    entry = SCENARIOS.get(scenario)
    if entry is None:
        print(f"Unknown ASR benchmark scenario: {scenario}\n", file=sys.stderr)
        print_usage()
        return 1

    _, module_name, _ = entry
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [f"benchmarks/asr {scenario}"] + argv[1:]
        result = module.main()
        return int(result) if result is not None else 0
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
