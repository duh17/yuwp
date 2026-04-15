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

MAIN_TREE="${YUWP_MAIN_TREE:-$PWD}"
SRC_METALLIB="$MAIN_TREE/.build/arm64-apple-macosx/release/mlx.metallib"
DST_METALLIB="$PWD/.build/arm64-apple-macosx/release/mlx.metallib"

mkdir -p "$(dirname "$DST_METALLIB")"
if [ "$SRC_METALLIB" != "$DST_METALLIB" ]; then
  if ! cp "$SRC_METALLIB" "$DST_METALLIB" >>"$build_log" 2>&1; then
    tail -80 "$build_log"
    exit 1
  fi
fi

if ! swift test >"$test_log" 2>&1; then
  tail -80 "$test_log"
  exit 1
fi
