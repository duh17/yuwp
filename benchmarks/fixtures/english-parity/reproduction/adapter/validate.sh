#!/bin/bash
# Build and model-free unit tests only. Never launch the adapter here.
set -euo pipefail
cd /tmp/yuwp-english-parity/adapter
export TMPDIR="$PWD/tmp"
export CLANG_MODULE_CACHE_PATH="$PWD/cache/clang"
export SWIFTPM_MODULECACHE_OVERRIDE="$PWD/cache/modules"
export XDG_CACHE_HOME="$PWD/cache"
export CFFIXED_USER_HOME="$PWD/cache/home"
mkdir -p "$TMPDIR" "$CFFIXED_USER_HOME" logs
SDK=/tmp/yuwp-english-parity/FluidAudio
test "$(git -C "$SDK" rev-parse HEAD)" = 5c19d5e12320e22bbfb7a1877b089d2665a69add
git -C "$SDK" diff --quiet
git -C "$SDK" diff --cached --quiet
swift build -c release --product fluid-parity-adapter --cache-path "$PWD/cache" --scratch-path "$PWD/.build" --disable-sandbox > logs/build-final.log 2>&1
swift test -c release --cache-path "$PWD/cache" --scratch-path "$PWD/.build" --disable-sandbox > logs/test-final.log 2>&1
{
    swift --version
    sw_vers
    printf 'FluidAudio revision: '; git -C "$SDK" rev-parse HEAD
    printf 'SDK source diff SHA256: '; git -C "$SDK" diff HEAD | shasum -a 256
    printf 'Package.resolved: absent (only local path dependency plus pinned binary artifact)\n'
} > logs/environment.txt 2>&1
shasum -a 256 Package.swift Sources/ASRIPC/Protocol.swift Sources/AdapterCore/Core.swift Sources/FluidAdapter/Backend.swift Sources/FluidAdapter/Main.swift Tests/AdapterCoreTests/AdapterTests.swift validate.sh > source-inventory.sha256
shasum -a 256 source-inventory.sha256 > source-tree.sha256
shasum -a 256 .build/arm64-apple-macosx/release/fluid-parity-adapter > binary.sha256
printf 'Build and model-free tests passed. No model load or inference executed.\n'
