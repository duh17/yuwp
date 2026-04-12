#!/usr/bin/env bash
# Build mlx.metallib from mlx-swift Metal kernel sources.
#
# swift build compiles Swift code but does NOT produce mlx.metallib.
# Without it, MLX fails at runtime: "Failed to load the default metallib."
#
# Usage: ./scripts/build_mlx_metallib.sh [release|debug]
#
# Prerequisites for a fresh build:
#   - Xcode Metal Toolchain (xcrun metal + metallib)
#   - swift build must have been run first (to resolve .build/checkouts/mlx-swift)
#
# Fallback mode:
#   - If Metal Toolchain is unavailable and YUWP_ALLOW_STALE_METALLIB=1,
#     reuse an existing mlx.metallib from the target configuration or copy one
#     from the other build configuration.
#
# Approach adapted from speech-swift (github.com/soniqo/speech-swift)

set -euo pipefail

CONFIGURATION="${1:-release}"
ALLOW_STALE_METALLIB="${YUWP_ALLOW_STALE_METALLIB:-0}"
BUILD_DIR=".build/arm64-apple-macosx/${CONFIGURATION}"
CHECKOUT_DIR=".build/checkouts/mlx-swift"
METAL_SRC_DIR="${CHECKOUT_DIR}/Source/Cmlx/mlx/mlx/backend/metal/kernels"
TARGET_METALLIB="${BUILD_DIR}/mlx.metallib"
HASH_FILE="${BUILD_DIR}/.mlx_metallib_hash"

if [ "${CONFIGURATION}" = "release" ]; then
    FALLBACK_CONFIGURATION="debug"
else
    FALLBACK_CONFIGURATION="release"
fi
FALLBACK_BUILD_DIR=".build/arm64-apple-macosx/${FALLBACK_CONFIGURATION}"
FALLBACK_METALLIB="${FALLBACK_BUILD_DIR}/mlx.metallib"
FALLBACK_HASH_FILE="${FALLBACK_BUILD_DIR}/.mlx_metallib_hash"

log() {
    echo "[yuwp] $*"
}

warn() {
    echo "[yuwp] Warning: $*" >&2
}

die() {
    echo "[yuwp] Error: $*" >&2
    exit 1
}

short_hash() {
    local value="${1:-}"
    if [ -z "${value}" ]; then
        echo "unknown"
    else
        echo "${value:0:8}"
    fi
}

current_hash() {
    find "${METAL_SRC_DIR}" -name "*.metal" | sort | xargs md5 -q 2>/dev/null | md5 -q || echo "unknown"
}

metal_toolchain_available() {
    xcrun --sdk macosx -f metallib >/dev/null 2>&1
}

reuse_existing_metallib() {
    local current_hash="$1"

    if [ -f "${TARGET_METALLIB}" ]; then
        local cached_hash=""
        if [ -f "${HASH_FILE}" ]; then
            cached_hash=$(cat "${HASH_FILE}")
        fi

        if [ -n "${cached_hash}" ] && [ "${cached_hash}" = "${current_hash}" ]; then
            log "mlx.metallib is up to date (hash: $(short_hash "${current_hash}"))"
        else
            warn "Metal Toolchain unavailable; reusing existing ${CONFIGURATION} mlx.metallib (cached: $(short_hash "${cached_hash}"), current: $(short_hash "${current_hash}"))"
        fi
        return 0
    fi

    if [ -f "${FALLBACK_METALLIB}" ]; then
        mkdir -p "${BUILD_DIR}"
        cp -f "${FALLBACK_METALLIB}" "${TARGET_METALLIB}"
        if [ -f "${FALLBACK_HASH_FILE}" ]; then
            cp -f "${FALLBACK_HASH_FILE}" "${HASH_FILE}"
        fi

        local fallback_hash=""
        if [ -f "${FALLBACK_HASH_FILE}" ]; then
            fallback_hash=$(cat "${FALLBACK_HASH_FILE}")
        fi

        warn "Metal Toolchain unavailable; copied ${FALLBACK_CONFIGURATION} mlx.metallib into ${CONFIGURATION} (cached: $(short_hash "${fallback_hash}"), current: $(short_hash "${current_hash}"))"
        return 0
    fi

    return 1
}

if [ ! -d "${METAL_SRC_DIR}" ]; then
    die "Metal kernel sources not found at ${METAL_SRC_DIR}. Run 'swift build' first to fetch dependencies."
fi

CURRENT_HASH=$(current_hash)

if [ -f "${HASH_FILE}" ] && [ -f "${TARGET_METALLIB}" ]; then
    CACHED_HASH=$(cat "${HASH_FILE}")
    if [ "${CURRENT_HASH}" = "${CACHED_HASH}" ]; then
        log "mlx.metallib is up to date (hash: $(short_hash "${CURRENT_HASH}"))"
        exit 0
    fi
fi

if ! metal_toolchain_available; then
    if [ "${ALLOW_STALE_METALLIB}" = "1" ]; then
        if reuse_existing_metallib "${CURRENT_HASH}"; then
            exit 0
        fi
        die "Metal Toolchain unavailable and no existing mlx.metallib could be reused. Install it with: xcodebuild -downloadComponent MetalToolchain"
    fi

    die "Metal Toolchain unavailable. Install it with: xcodebuild -downloadComponent MetalToolchain or rerun with YUWP_ALLOW_STALE_METALLIB=1 to reuse an existing mlx.metallib."
fi

log "Building mlx.metallib from Metal kernel sources..."

AIR_DIR=$(mktemp -d)
trap 'rm -rf "${AIR_DIR}"' EXIT

SDK_VERSION=$(xcrun --sdk macosx --show-sdk-version)

METAL_FILES=()
AIR_FILES=()
while IFS= read -r -d '' metalfile; do
    METAL_FILES+=("${metalfile}")
done < <(find "${METAL_SRC_DIR}" -name "*.metal" -print0)

if [ ${#METAL_FILES[@]} -eq 0 ]; then
    die "No .metal files found in ${METAL_SRC_DIR}"
fi

log "Compiling ${#METAL_FILES[@]} Metal files..."

for metalfile in "${METAL_FILES[@]}"; do
    basename=$(basename "${metalfile}" .metal)
    airfile="${AIR_DIR}/${basename}.air"
    if xcrun -sdk macosx metal \
        -c "${metalfile}" \
        -o "${airfile}" \
        -I "${CHECKOUT_DIR}/Source/Cmlx/mlx" \
        -target "air64-apple-macosx${SDK_VERSION}" \
        2>/dev/null; then
        AIR_FILES+=("${airfile}")
    else
        warn "Failed to compile ${basename}.metal, skipping..."
        rm -f "${airfile}"
    fi
done

if [ ${#AIR_FILES[@]} -eq 0 ]; then
    die "Failed to compile any Metal kernels into .air files"
fi

log "Linking mlx.metallib..."
mkdir -p "${BUILD_DIR}"
xcrun -sdk macosx metallib \
    "${AIR_FILES[@]}" \
    -o "${TARGET_METALLIB}"

echo "${CURRENT_HASH}" > "${HASH_FILE}"

log "Successfully built ${TARGET_METALLIB}"
log "  Size: $(du -sh "${TARGET_METALLIB}" | cut -f1)"
