#!/bin/bash
# Build Yuwp, yuwp-asr, yuwp-tts, and compile mlx.metallib.
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

if [ ${#SWIFT_FLAGS[@]} -gt 0 ]; then
    swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product Yuwp
    swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product yuwp-asr
    swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product yuwp-tts
else
    swift build -c "$CONFIGURATION" --product Yuwp
    swift build -c "$CONFIGURATION" --product yuwp-asr
    swift build -c "$CONFIGURATION" --product yuwp-tts
fi
bash scripts/build_mlx_metallib.sh "$CONFIGURATION"

echo "[yuwp] Built Yuwp + yuwp-asr + yuwp-tts ($CONFIGURATION)"
echo "[yuwp] Binaries: .build/arm64-apple-macosx/$CONFIGURATION/{Yuwp,yuwp-asr,yuwp-tts,mlx.metallib}"
