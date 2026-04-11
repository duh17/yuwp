#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

build_log=$(mktemp)
test_log=$(mktemp)
trap 'rm -f "$build_log" "$test_log"' EXIT

if ! swift build -c release --product asr-stream-test >"$build_log" 2>&1; then
  tail -80 "$build_log"
  exit 1
fi

MAIN_TREE="/Users/chenda/workspace/yuwp"
mkdir -p .build/arm64-apple-macosx/release
if ! cp "$MAIN_TREE/.build/arm64-apple-macosx/release/mlx.metallib" .build/arm64-apple-macosx/release/mlx.metallib >>"$build_log" 2>&1; then
  tail -80 "$build_log"
  exit 1
fi

if ! swift test >"$test_log" 2>&1; then
  tail -80 "$test_log"
  exit 1
fi
