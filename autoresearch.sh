#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

swift build -c release --product asr-stream-test >/dev/null
MAIN_TREE="/Users/chenda/workspace/yuwp"
mkdir -p .build/arm64-apple-macosx/release
cp "$MAIN_TREE/.build/arm64-apple-macosx/release/mlx.metallib" .build/arm64-apple-macosx/release/mlx.metallib
python3 scripts/bench-stream-quality.py
