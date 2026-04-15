#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

swift build -c release --product asr-stream-test >/dev/null

MAIN_TREE="${YUWP_MAIN_TREE:-$PWD}"
SRC_METALLIB="$MAIN_TREE/.build/arm64-apple-macosx/release/mlx.metallib"
DST_METALLIB="$PWD/.build/arm64-apple-macosx/release/mlx.metallib"

mkdir -p "$(dirname "$DST_METALLIB")"
if [ "$SRC_METALLIB" != "$DST_METALLIB" ]; then
  cp "$SRC_METALLIB" "$DST_METALLIB"
fi

python3 scripts/bench-stream-quality.py
