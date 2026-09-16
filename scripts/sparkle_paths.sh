#!/bin/bash
# Resolve Sparkle.framework and Sparkle CLI tools for packaging scripts.
#
# Swift 6.4 Swift Build copies Sparkle.framework next to Yuwp under
# `swift build --show-bin-path`. Older native SwiftPM left the xcframework
# under `.build/artifacts/sparkle/...`. Prefer Products, then artifacts.
#
# Usage (from repo root, after BIN_DIR is known):
#   source scripts/sparkle_paths.sh
#   yuwp_resolve_sparkle_paths "$BIN_DIR" "$(pwd)"
# Sets SPARKLE_FW, SPARKLE_SIGN_UPDATE, SPARKLE_GENERATE_KEYS (empty if missing).

yuwp_sparkle_framework_candidates() {
    local bin_dir="$1"
    local root="$2"
    printf '%s\n' \
        "$bin_dir/Sparkle.framework" \
        "$root/.build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"
}

yuwp_sparkle_tool_candidates() {
    local tool="$1"
    local bin_dir="$2"
    local root="$3"
    printf '%s\n' \
        "$bin_dir/$tool" \
        "$root/.build/artifacts/sparkle/Sparkle/bin/$tool"
}

yuwp_first_existing_dir() {
    local candidate
    while IFS= read -r candidate; do
        [ -n "$candidate" ] || continue
        if [ -d "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

yuwp_first_executable() {
    local candidate
    while IFS= read -r candidate; do
        [ -n "$candidate" ] || continue
        if [ -f "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

yuwp_resolve_sparkle_paths() {
    local bin_dir="$1"
    local root="$2"
    SPARKLE_FW="$(yuwp_sparkle_framework_candidates "$bin_dir" "$root" | yuwp_first_existing_dir || true)"
    SPARKLE_SIGN_UPDATE="$(yuwp_sparkle_tool_candidates "sign_update" "$bin_dir" "$root" | yuwp_first_executable || true)"
    SPARKLE_GENERATE_KEYS="$(yuwp_sparkle_tool_candidates "generate_keys" "$bin_dir" "$root" | yuwp_first_executable || true)"
}
