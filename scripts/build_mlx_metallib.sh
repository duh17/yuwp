#!/usr/bin/env bash
# Build mlx.metallib from mlx-swift Metal kernel sources.
#
# swift build compiles Swift code but does NOT produce mlx.metallib.
# Without it, MLX fails at runtime: "Failed to load the default metallib."
#
# Usage: ./scripts/build_mlx_metallib.sh [release|debug]
#
# Prerequisites:
#   - Xcode Metal Toolchain (xcrun metal must be available)
#   - swift build must have been run first (to resolve .build/checkouts/mlx-swift)
#
# Approach adapted from speech-swift (github.com/soniqo/speech-swift)

set -euo pipefail

CONFIGURATION="${1:-release}"
BUILD_DIR=".build/arm64-apple-macosx/${CONFIGURATION}"
CHECKOUT_DIR=".build/checkouts/mlx-swift"

# Locate Metal kernel sources
METAL_SRC_DIR="${CHECKOUT_DIR}/Source/Cmlx/mlx/mlx/backend/metal/kernels"

if [ ! -d "${METAL_SRC_DIR}" ]; then
    echo "Error: Metal kernel sources not found at ${METAL_SRC_DIR}"
    echo "Run 'swift build' first to fetch dependencies."
    exit 1
fi

# Content hash for cache invalidation
HASH_FILE="${BUILD_DIR}/.mlx_metallib_hash"
CURRENT_HASH=$(find "${METAL_SRC_DIR}" -name "*.metal" | sort | xargs md5 -q 2>/dev/null | md5 -q || echo "unknown")

if [ -f "${HASH_FILE}" ] && [ -f "${BUILD_DIR}/mlx.metallib" ]; then
    CACHED_HASH=$(cat "${HASH_FILE}")
    if [ "${CURRENT_HASH}" = "${CACHED_HASH}" ]; then
        echo "mlx.metallib is up to date (hash: ${CURRENT_HASH:0:8})"
        exit 0
    fi
fi

echo "Building mlx.metallib from Metal kernel sources..."

# Temp directory for .air files
AIR_DIR=$(mktemp -d)
trap "rm -rf ${AIR_DIR}" EXIT

# Get SDK path and version
SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
SDK_VERSION=$(xcrun --sdk macosx --show-sdk-version)

# Compile each .metal file to .air
METAL_FILES=()
AIR_FILES=()

while IFS= read -r -d '' metalfile; do
    METAL_FILES+=("${metalfile}")
done < <(find "${METAL_SRC_DIR}" -name "*.metal" -print0)

if [ ${#METAL_FILES[@]} -eq 0 ]; then
    echo "Error: No .metal files found in ${METAL_SRC_DIR}"
    exit 1
fi

echo "Compiling ${#METAL_FILES[@]} Metal files..."

for metalfile in "${METAL_FILES[@]}"; do
    basename=$(basename "${metalfile}" .metal)
    airfile="${AIR_DIR}/${basename}.air"
    AIR_FILES+=("${airfile}")
    xcrun -sdk macosx metal \
        -c "${metalfile}" \
        -o "${airfile}" \
        -I "${CHECKOUT_DIR}/Source/Cmlx/mlx" \
        -target "air64-apple-macosx${SDK_VERSION}" \
        2>/dev/null || {
        echo "Warning: Failed to compile ${basename}.metal, skipping..."
        # Remove failed .air so metallib doesn't get a broken file
        rm -f "${airfile}"
        continue
    }
done

# Link all .air files into mlx.metallib
echo "Linking mlx.metallib..."
xcrun -sdk macosx metallib \
    "${AIR_FILES[@]}" \
    -o "${BUILD_DIR}/mlx.metallib"

# Cache hash
echo "${CURRENT_HASH}" > "${HASH_FILE}"

echo "Successfully built ${BUILD_DIR}/mlx.metallib"
echo "  Size: $(du -sh "${BUILD_DIR}/mlx.metallib" | cut -f1)"
