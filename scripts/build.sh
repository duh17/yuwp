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

# Swift 6.4 SwiftPM defaults to Swift Build, which compiles mlx-swift Metal
# sources and fails unless the standalone Metal toolchain is installed.
# Native keeps .build/arm64-apple-macosx and the separate mlx.metallib script.
swift_build() {
    swift build --build-system native "$@"
}

if [ ${#SWIFT_FLAGS[@]} -gt 0 ]; then
    swift_build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product Yuwp
    swift_build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product yuwp-asr
    swift_build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product yuwp-tts
else
    swift_build -c "$CONFIGURATION" --product Yuwp
    swift_build -c "$CONFIGURATION" --product yuwp-asr
    swift_build -c "$CONFIGURATION" --product yuwp-tts
fi
bash scripts/build_mlx_metallib.sh "$CONFIGURATION"

echo "[yuwp] Built Yuwp + yuwp-asr + yuwp-tts ($CONFIGURATION)"
echo "[yuwp] Binaries: .build/arm64-apple-macosx/$CONFIGURATION/{Yuwp,yuwp-asr,yuwp-tts,mlx.metallib}"
