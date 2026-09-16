#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# dependencies = ["numpy", "psutil", "requests", "matplotlib"]
# ///
"""Front door for repo-local benchmark tooling."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

COMMANDS = {
    "asr": ("lib.asr", "Unified ASR benchmark scenarios: headline, load, quality, compare, remote, subtitles"),
    "plot": ("lib.plot_native", "Plot benchmark JSON from `asr load --json`"),
}

def print_usage() -> None:
    print("Usage: benchmarks/cli.py <command> [options]\n")
    print("Commands:")
    for name, (_, summary) in COMMANDS.items():
        print(f"  {name:<15} {summary}")


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"--help", "-h", "help"}:
        print_usage()
        return 0

    command_name = argv[0]
    command_args = argv[1:]
    command = COMMANDS.get(command_name)
    if command is None:
        print(f"Unknown benchmark command: {command_name}\n", file=sys.stderr)
        print_usage()
        return 1

    module_name, _ = command
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [f"benchmarks/{command_name}"] + command_args
        result = module.main()
        return int(result) if result is not None else 0
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
