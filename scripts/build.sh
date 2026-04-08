#!/bin/bash
# Build Yuwp + native asr-server and compile mlx.metallib.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGURATION="${1:-release}"

swift build -c "$CONFIGURATION" --product Yuwp --product asr-server
bash scripts/build_mlx_metallib.sh "$CONFIGURATION"

echo "[yuwp] Built Yuwp + asr-server ($CONFIGURATION)"
echo "[yuwp] Binaries: .build/arm64-apple-macosx/$CONFIGURATION/{Yuwp,asr-server,mlx.metallib}"
