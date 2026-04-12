#!/bin/bash
# Build Yuwp + native ASR binaries (`asr-server`, `yuwp-asr`) and compile mlx.metallib.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGURATION="${1:-release}"
ENABLE_INTERNAL_DIAGNOSTICS="${YUWP_INTERNAL_DIAGNOSTICS:-0}"
: "${YUWP_ALLOW_STALE_METALLIB:=1}"
export YUWP_ALLOW_STALE_METALLIB

SWIFT_FLAGS=()
if [ "$ENABLE_INTERNAL_DIAGNOSTICS" = "1" ]; then
    SWIFT_FLAGS+=("-Xswiftc" "-DYUWP_INTERNAL_DIAGNOSTICS")
    echo "[yuwp] Internal diagnostics: ON"
else
    echo "[yuwp] Internal diagnostics: OFF"
fi

echo "[yuwp] Allow stale metallib: $YUWP_ALLOW_STALE_METALLIB"

swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product Yuwp
swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product asr-server
swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product yuwp-asr
bash scripts/build_mlx_metallib.sh "$CONFIGURATION"

echo "[yuwp] Built Yuwp + asr-server + yuwp-asr ($CONFIGURATION)"
echo "[yuwp] Binaries: .build/arm64-apple-macosx/$CONFIGURATION/{Yuwp,asr-server,yuwp-asr,mlx.metallib}"
