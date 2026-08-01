#!/bin/bash
set -euo pipefail

# Correctness checks for autoresearch: fresh release builds, unit tests, and
# model-backed batch/streaming quality gates.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TMP_DIR=$(mktemp -d /tmp/yuwp-autoresearch-checks.XXXXXX)
trap 'rm -rf "$TMP_DIR"' EXIT

build_product() {
    local product="$1"
    local log="$TMP_DIR/build-$product.log"
    echo "=== swift build -c release --product $product ===" >&2
    if ! swift build -c release --product "$product" >"$log" 2>&1; then
        cat "$log" >&2
        echo "FAIL: release build failed for $product" >&2
        exit 1
    fi
    grep -iE "error|warning" "$log" >&2 || true
}

build_product yuwp-asr
build_product asr-stream-test

SERVER_BIN=".build/arm64-apple-macosx/release/yuwp-asr"
STREAM_BIN=".build/arm64-apple-macosx/release/asr-stream-test"
for binary in "$SERVER_BIN" "$STREAM_BIN"; do
    if [[ ! -x "$binary" ]]; then
        echo "FAIL: release binary not produced: $binary" >&2
        exit 1
    fi
done

# Unit and integration-free behavior tests.
echo "=== swift test ===" >&2
if ! swift test >"$TMP_DIR/swift-test.log" 2>&1; then
    tail -100 "$TMP_DIR/swift-test.log" >&2
    echo "FAIL: swift test failed" >&2
    exit 1
fi
tail -5 "$TMP_DIR/swift-test.log"

MODEL="$HOME/Library/Application Support/Yuwp/models/mlx-community--Qwen3-ASR-0.6B-bf16"
if [[ ! -d "$MODEL" ]]; then
    echo "FAIL: quality-gate model not found: $MODEL" >&2
    exit 1
fi

transcribe_text() {
    local fixture="$1"
    local language="$2"
    "$SERVER_BIN" transcribe "$fixture" --model "$MODEL" --format json --language "$language" 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['text'])"
}

check_exact_transcript() {
    local fixture="$1"
    local language="$2"
    local golden="$3"
    local actual expected
    actual=$(transcribe_text "$fixture" "$language")
    expected=$(tr -d '\n' <"$golden")
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL: transcript quality regression for $fixture" >&2
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        exit 1
    fi
}

check_exact_transcript Tests/fixtures/jfk.wav English .auto/golden_en.txt
check_exact_transcript Tests/fixtures/asr_zh.wav Chinese .auto/golden_zh.txt

check_stream_vs_batch_wer() {
    local fixture="$1"
    local max_wer="$2"
    local report
    if ! report=$("$STREAM_BIN" "$fixture" --model "$MODEL" \
        --no-batch-retranscribe --compact 2>/dev/null); then
        echo "FAIL: core streaming quality run failed for $fixture" >&2
        exit 1
    fi
    STREAM_REPORT="$report" python3 - "$fixture" "$max_wer" <<'PY'
import json
import os
import sys

fixture, maximum = sys.argv[1], float(sys.argv[2])
report = json.loads(os.environ["STREAM_REPORT"])
accuracy = report.get("accuracy")
if accuracy is None:
    raise SystemExit(f"FAIL: no accuracy report for {fixture}")
wer = float(accuracy["wordErrorRate"])
if wer > maximum:
    raise SystemExit(
        f"FAIL: stream-vs-batch WER for {fixture} is {wer:.4f}; maximum is {maximum:.4f}"
    )
print(f"streaming quality: {fixture} stream-vs-batch WER={wer:.4f} (max {maximum:.4f})")
PY
}

# Core streaming is tested without the batch-correction fallback so cache and
# prefill regressions cannot be hidden by a final batch retranscription.
check_stream_vs_batch_wer Tests/fixtures/jfk.wav 0.01
check_stream_vs_batch_wer Tests/fixtures/asr_en.wav 0.07

echo "All checks passed."
