#!/bin/bash
# Build Yuwp + native asr-server and compile mlx.metallib.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGURATION="${1:-release}"
ENABLE_INTERNAL_DIAGNOSTICS="${YUWP_INTERNAL_DIAGNOSTICS:-0}"

SWIFT_FLAGS=()
if [ "$ENABLE_INTERNAL_DIAGNOSTICS" = "1" ]; then
    SWIFT_FLAGS+=("-Xswiftc" "-DYUWP_INTERNAL_DIAGNOSTICS")
    echo "[yuwp] Internal diagnostics: ON"
else
    echo "[yuwp] Internal diagnostics: OFF"
fi

swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product Yuwp
swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product asr-server
bash scripts/build_mlx_metallib.sh "$CONFIGURATION"

echo "[yuwp] Built Yuwp + asr-server ($CONFIGURATION)"
echo "[yuwp] Binaries: .build/arm64-apple-macosx/$CONFIGURATION/{Yuwp,asr-server,mlx.metallib}"
