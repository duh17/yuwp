#!/bin/bash
# Focused tests for Sparkle.framework / sign_update discovery.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=sparkle_paths.sh
source "$ROOT/scripts/sparkle_paths.sh"

fail=0

assert_eq() {
    local label="$1"
    local expected="$2"
    local actual="$3"
    if [ "$expected" != "$actual" ]; then
        echo "FAIL: $label"
        echo "  expected: $expected"
        echo "  actual:   $actual"
        fail=$((fail + 1))
    else
        echo "PASS: $label"
    fi
}

assert_empty() {
    local label="$1"
    local actual="$2"
    if [ -n "$actual" ]; then
        echo "FAIL: $label"
        echo "  expected empty, got: $actual"
        fail=$((fail + 1))
    else
        echo "PASS: $label"
    fi
}

TMP="$(mktemp -d /tmp/yuwp-sparkle-paths.XXXXXX)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

PRODUCTS="$TMP/.build/out/Products/Release"
ARTIFACTS_FW_DIR="$TMP/.build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64"
ARTIFACTS_BIN="$TMP/.build/artifacts/sparkle/Sparkle/bin"

# Candidate order prefers Swift Build Products, then the native artifacts path.
FW_CANDIDATES=()
while IFS= read -r line; do
    FW_CANDIDATES+=("$line")
done < <(yuwp_sparkle_framework_candidates "$PRODUCTS" "$TMP")
assert_eq "framework candidate 0 is Products Sparkle.framework" \
    "$PRODUCTS/Sparkle.framework" "${FW_CANDIDATES[0]:-}"
assert_eq "framework candidate 1 is artifacts xcframework slice" \
    "$ARTIFACTS_FW_DIR/Sparkle.framework" "${FW_CANDIDATES[1]:-}"
assert_eq "framework candidate count" "2" "${#FW_CANDIDATES[@]}"

TOOL_CANDIDATES=()
while IFS= read -r line; do
    TOOL_CANDIDATES+=("$line")
done < <(yuwp_sparkle_tool_candidates "sign_update" "$PRODUCTS" "$TMP")
assert_eq "sign_update candidate 0 is Products" \
    "$PRODUCTS/sign_update" "${TOOL_CANDIDATES[0]:-}"
assert_eq "sign_update candidate 1 is artifacts/bin" \
    "$ARTIFACTS_BIN/sign_update" "${TOOL_CANDIDATES[1]:-}"
assert_eq "sign_update candidate count" "2" "${#TOOL_CANDIDATES[@]}"

# Neither layout present → empty resolution.
yuwp_resolve_sparkle_paths "$PRODUCTS" "$TMP"
assert_empty "framework missing when neither layout exists" "${SPARKLE_FW:-}"
assert_empty "sign_update missing when neither layout exists" "${SPARKLE_SIGN_UPDATE:-}"
assert_empty "generate_keys missing when neither layout exists" "${SPARKLE_GENERATE_KEYS:-}"

# Artifacts-only fallback (old native SwiftPM layout).
mkdir -p "$ARTIFACTS_FW_DIR/Sparkle.framework" "$ARTIFACTS_BIN"
touch "$ARTIFACTS_BIN/sign_update" "$ARTIFACTS_BIN/generate_keys"
chmod +x "$ARTIFACTS_BIN/sign_update" "$ARTIFACTS_BIN/generate_keys"
yuwp_resolve_sparkle_paths "$PRODUCTS" "$TMP"
assert_eq "framework falls back to artifacts xcframework" \
    "$ARTIFACTS_FW_DIR/Sparkle.framework" "${SPARKLE_FW:-}"
assert_eq "sign_update falls back to artifacts/bin" \
    "$ARTIFACTS_BIN/sign_update" "${SPARKLE_SIGN_UPDATE:-}"
assert_eq "generate_keys falls back to artifacts/bin" \
    "$ARTIFACTS_BIN/generate_keys" "${SPARKLE_GENERATE_KEYS:-}"

# Products layout wins when both exist.
mkdir -p "$PRODUCTS/Sparkle.framework"
touch "$PRODUCTS/sign_update" "$PRODUCTS/generate_keys"
chmod +x "$PRODUCTS/sign_update" "$PRODUCTS/generate_keys"
yuwp_resolve_sparkle_paths "$PRODUCTS" "$TMP"
assert_eq "framework prefers Products Sparkle.framework" \
    "$PRODUCTS/Sparkle.framework" "${SPARKLE_FW:-}"
assert_eq "sign_update prefers Products" \
    "$PRODUCTS/sign_update" "${SPARKLE_SIGN_UPDATE:-}"
assert_eq "generate_keys prefers Products" \
    "$PRODUCTS/generate_keys" "${SPARKLE_GENERATE_KEYS:-}"

# Packaging scripts keep the same public Sparkle defaults.
extract_default() {
    local key="$1" file="$2"
    sed -n "s/^${key}=\"\(.*\)\".*/\1/p" "$file" | head -n 1
}
RUN_FEED=$(extract_default DEFAULT_SPARKLE_FEED_URL "$ROOT/scripts/run.sh")
REL_FEED=$(extract_default DEFAULT_SPARKLE_FEED_URL "$ROOT/scripts/release.sh")
RUN_KEY=$(extract_default DEFAULT_SPARKLE_PUBLIC_ED_KEY "$ROOT/scripts/run.sh")
REL_KEY=$(extract_default DEFAULT_SPARKLE_PUBLIC_ED_KEY "$ROOT/scripts/release.sh")
assert_eq "run.sh keeps default Sparkle feed URL" \
    "https://github.com/duh17/yuwp/releases/latest/download/appcast.xml" "$RUN_FEED"
assert_eq "release.sh feed URL matches run.sh" "$RUN_FEED" "$REL_FEED"
if [ -z "$RUN_KEY" ]; then
    echo "FAIL: run.sh missing DEFAULT_SPARKLE_PUBLIC_ED_KEY"
    fail=$((fail + 1))
elif [ "$RUN_KEY" = "$REL_KEY" ]; then
    echo "PASS: release.sh public Ed key matches run.sh"
else
    echo "FAIL: release.sh public Ed key does not match run.sh"
    fail=$((fail + 1))
fi

# Packaging scripts must not hardcode the native-only artifact path as the sole lookup.
if grep -n 'SPARKLE_FW="\.build/artifacts/sparkle' "$ROOT/scripts/run.sh" "$ROOT/scripts/release.sh"; then
    echo "FAIL: run.sh/release.sh still hardcode SPARKLE_FW to the native-only artifacts path"
    fail=$((fail + 1))
else
    echo "PASS: run.sh/release.sh do not hardcode native-only SPARKLE_FW"
fi
if grep -n 'SPARKLE_TOOLS_DIR="\.build/artifacts/sparkle/Sparkle/bin"' "$ROOT/scripts/release.sh"; then
    echo "FAIL: release.sh still hardcodes SPARKLE_TOOLS_DIR to the native-only artifacts path"
    fail=$((fail + 1))
else
    echo "PASS: release.sh does not hardcode native-only SPARKLE_TOOLS_DIR"
fi

if [ "$fail" -ne 0 ]; then
    echo "$fail test(s) failed"
    exit 1
fi
echo "All sparkle path tests passed"
