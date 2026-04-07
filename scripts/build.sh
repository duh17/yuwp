#!/bin/bash
# Build Yuwp and codesign with stable identifier.
# Accessibility permission is tied to code signing hash — without a
# stable identifier, every `swift build` revokes the permission.
set -euo pipefail

cd "$(dirname "$0")/.."
swift build "$@"
codesign --force --sign - --identifier "com.yuwp.app" .build/debug/yuwp
echo "[yuwp] Signed as com.yuwp.app"
