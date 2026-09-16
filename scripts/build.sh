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
# into mlx-swift_Cmlx.bundle. Requires: xcodebuild -downloadComponent MetalToolchain
if [ ${#SWIFT_FLAGS[@]} -gt 0 ]; then
    swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product Yuwp
    swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product yuwp-asr
    swift build -c "$CONFIGURATION" "${SWIFT_FLAGS[@]}" --product yuwp-tts
else
    swift build -c "$CONFIGURATION" --product Yuwp
    swift build -c "$CONFIGURATION" --product yuwp-asr
    swift build -c "$CONFIGURATION" --product yuwp-tts
fi

BIN_DIR=$(swift build --show-bin-path -c "$CONFIGURATION")
SWIFTPM_METALLIB="$BIN_DIR/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
if [ -f "$SWIFTPM_METALLIB" ]; then
    cp -f "$SWIFTPM_METALLIB" "$BIN_DIR/mlx.metallib"
    echo "[yuwp] Installed mlx.metallib from SwiftPM Cmlx bundle"
else
    echo "[yuwp] SwiftPM Cmlx metallib missing; compiling kernels"
    YUWP_BIN_DIR="$BIN_DIR" bash scripts/build_mlx_metallib.sh "$CONFIGURATION"
fi

echo "[yuwp] Built Yuwp + yuwp-asr + yuwp-tts ($CONFIGURATION)"
echo "[yuwp] Binaries: $BIN_DIR/{Yuwp,yuwp-asr,yuwp-tts,mlx.metallib}"
