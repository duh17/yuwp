#!/bin/bash
set -euo pipefail

# Correctness checks for autoresearch: build, tests, quality gate.
# Only errors are shown — success output is suppressed.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 1. Release build (also needed by measure.sh)
echo "=== swift build -c release ===" >&2
swift build -c release --product yuwp-asr 2>&1 | grep -iE "error|warning" || true
if [[ ! -x .build/arm64-apple-macosx/release/yuwp-asr ]]; then
    echo "FAIL: release binary not produced"
    exit 1
fi

# 2. Test suite
echo "=== swift test ===" >&2
swift test 2>&1 | tail -5
SWIFT_TEST_EXIT=${PIPESTATUS[0]}
if [[ $SWIFT_TEST_EXIT -ne 0 ]]; then
    echo "FAIL: swift test exited $SWIFT_TEST_EXIT"
    exit 1
fi

# 3. Quality gate: jfk.wav transcript must match golden (exact match)
SERVER_BIN=".build/arm64-apple-macosx/release/yuwp-asr"
MODEL="$HOME/Library/Application Support/Yuwp/models/mlx-community--Qwen3-ASR-0.6B-bf16"
GOLDEN=".auto/golden_en.txt"

ACTUAL=$("$SERVER_BIN" transcribe Tests/fixtures/jfk.wav --model "$MODEL" --format json --language English 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])")

EXPECTED=$(cat "$GOLDEN" | tr -d '\n')

if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo "FAIL: transcript quality regression"
    echo "  expected: $EXPECTED"
    echo "  actual:   $ACTUAL"
    exit 1
fi

echo "All checks passed."
