#!/bin/bash
# Build, sign, notarize, and package a Yuwp release.
#
# Usage: release.sh <version>
#   e.g.: release.sh 0.2.0
#
# Required environment variables:
#   YUWP_SIGN_IDENTITY      — Developer ID Application identity
#   YUWP_NOTARY_PROFILE     — optional notarytool keychain profile name
#   YUWP_TEAM_ID            — Apple Team ID (required when not using YUWP_NOTARY_PROFILE)
#   YUWP_APPLE_ID           — Apple ID email for notarytool (required when not using YUWP_NOTARY_PROFILE)
#   YUWP_APP_PASSWORD       — App-specific password for notarytool (required when not using YUWP_NOTARY_PROFILE)
#   YUWP_SPARKLE_FEED_URL   — optional Sparkle appcast URL override
#   YUWP_SPARKLE_PUBLIC_ED_KEY — optional Sparkle public key override
#
# Output:
#   release/Yuwp-<version>.dmg  — notarized, stapled disk image
#   release/appcast.xml         — Sparkle appcast entry
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION="${1:?Usage: release.sh <version>}"
SIGN_IDENTITY="${YUWP_SIGN_IDENTITY:?Set YUWP_SIGN_IDENTITY}"
NOTARY_PROFILE="${YUWP_NOTARY_PROFILE:-}"
DEFAULT_SPARKLE_FEED_URL="https://github.com/duh17/yuwp/releases/latest/download/appcast.xml"
DEFAULT_SPARKLE_PUBLIC_ED_KEY="wnLCIfY048anOcj7/J/Iv6Lp9Fmba4zQ0EjCL7k/M+E=" # gitleaks:allow public Sparkle key
SPARKLE_FEED_URL="${YUWP_SPARKLE_FEED_URL:-$DEFAULT_SPARKLE_FEED_URL}"
SPARKLE_PUBLIC_ED_KEY="${YUWP_SPARKLE_PUBLIC_ED_KEY:-$DEFAULT_SPARKLE_PUBLIC_ED_KEY}"

if [ -z "$NOTARY_PROFILE" ]; then
    TEAM_ID="${YUWP_TEAM_ID:?Set YUWP_TEAM_ID or YUWP_NOTARY_PROFILE}"
    APPLE_ID="${YUWP_APPLE_ID:?Set YUWP_APPLE_ID or YUWP_NOTARY_PROFILE}"
    APP_PASSWORD="${YUWP_APP_PASSWORD:?Set YUWP_APP_PASSWORD or YUWP_NOTARY_PROFILE}"
fi

CONFIGURATION="release"
export YUWP_INTERNAL_DIAGNOSTICS=0
export YUWP_ALLOW_STALE_METALLIB=0
BIN_DIR=".build/arm64-apple-macosx/$CONFIGURATION"
RELEASE_DIR="release"
APP="$RELEASE_DIR/Yuwp.app"
MACOS_DIR="$APP/Contents/MacOS"
RES_DIR="$APP/Contents/Resources"
FRAMEWORKS_DIR="$APP/Contents/Frameworks"
DMG="$RELEASE_DIR/Yuwp-$VERSION.dmg"

SPARKLE_FW=".build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"
SPARKLE_TOOLS_DIR=$(find .build/artifacts -name "sign_update" -exec dirname {} \; 2>/dev/null | head -1)

# ── Build ──────────────────────────────────────────────────────────────
echo "=== Building Yuwp $VERSION ==="
bash scripts/build.sh "$CONFIGURATION"

# ── Assemble Bundle ────────────────────────────────────────────────────
echo "=== Assembling app bundle ==="
rm -rf "$RELEASE_DIR"
mkdir -p "$MACOS_DIR" "$RES_DIR" "$FRAMEWORKS_DIR"

cp -f "$BIN_DIR/Yuwp" "$MACOS_DIR/Yuwp"
cp -f "$BIN_DIR/asr-server" "$MACOS_DIR/asr-server"
cp -f "$BIN_DIR/mlx.metallib" "$MACOS_DIR/mlx.metallib"
cp -f "icon-layers/Yuwp.icns" "$RES_DIR/Yuwp.icns"

# SwiftPM doesn't add the app-bundle Frameworks runpath for this executable.
# Add it here so the packaged app can load Sparkle.framework at runtime.
install_name_tool -add_rpath "@executable_path/../Frameworks" "$MACOS_DIR/Yuwp"

if [ ! -d "$SPARKLE_FW" ]; then
    echo "Error: Sparkle.framework not found. Run 'swift package resolve' first."
    exit 1
fi
ditto "$SPARKLE_FW" "$FRAMEWORKS_DIR/Sparkle.framework"

cat > "$APP/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.yuwp.app</string>
    <key>CFBundleName</key>
    <string>Yuwp</string>
    <key>CFBundleDisplayName</key>
    <string>Yuwp</string>
    <key>CFBundleExecutable</key>
    <string>Yuwp</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>Yuwp</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Yuwp needs microphone access to transcribe your speech into text.</string>
    <key>SUFeedURL</key>
    <string>$SPARKLE_FEED_URL</string>
    <key>SUPublicEDKey</key>
    <string>$SPARKLE_PUBLIC_ED_KEY</string>
</dict>
</plist>
PLIST

# ── Code Sign ──────────────────────────────────────────────────────────
echo "=== Code signing ==="

# Sparkle helpers + framework (inside-out)
SPARKLE_VERSION_DIR="$FRAMEWORKS_DIR/Sparkle.framework/Versions/B"
for xpc in "$SPARKLE_VERSION_DIR"/XPCServices/*.xpc; do
    [ -e "$xpc" ] && codesign --force --sign "$SIGN_IDENTITY" --options runtime "$xpc"
done
if [ -d "$SPARKLE_VERSION_DIR/Updater.app" ]; then
    codesign --force --sign "$SIGN_IDENTITY" --options runtime "$SPARKLE_VERSION_DIR/Updater.app"
fi
if [ -f "$SPARKLE_VERSION_DIR/Autoupdate" ]; then
    codesign --force --sign "$SIGN_IDENTITY" --options runtime "$SPARKLE_VERSION_DIR/Autoupdate"
fi

# Sparkle framework
codesign --force --sign "$SIGN_IDENTITY" --options runtime \
    "$FRAMEWORKS_DIR/Sparkle.framework"

# Metal library
codesign --force --sign "$SIGN_IDENTITY" --options runtime \
    --identifier com.yuwp.app.metallib "$MACOS_DIR/mlx.metallib"

# asr-server (needs allow-unsigned-executable-memory for MLX)
codesign --force --sign "$SIGN_IDENTITY" --options runtime \
    --entitlements asr-server.entitlements \
    --identifier com.yuwp.app.server "$MACOS_DIR/asr-server"

# Main binary
codesign --force --sign "$SIGN_IDENTITY" --options runtime \
    --entitlements Yuwp.entitlements \
    --identifier com.yuwp.app "$MACOS_DIR/Yuwp"

# App bundle
codesign --force --sign "$SIGN_IDENTITY" --options runtime \
    --entitlements Yuwp.entitlements \
    --identifier com.yuwp.app "$APP"

# Verify
echo "=== Verifying signature ==="
codesign --verify --deep --strict "$APP"

# ── Create DMG ─────────────────────────────────────────────────────────
echo "=== Creating DMG ==="
hdiutil create -volname "Yuwp" -srcfolder "$APP" \
    -ov -format UDZO "$DMG"

codesign --force --sign "$SIGN_IDENTITY" "$DMG"

# ── Notarize ───────────────────────────────────────────────────────────
echo "=== Notarizing (this may take several minutes) ==="
if [ -n "$NOTARY_PROFILE" ]; then
    xcrun notarytool submit "$DMG" \
        --keychain-profile "$NOTARY_PROFILE" \
        --wait
else
    xcrun notarytool submit "$DMG" \
        --apple-id "$APPLE_ID" \
        --team-id "$TEAM_ID" \
        --password "$APP_PASSWORD" \
        --wait
fi

echo "=== Stapling ==="
xcrun stapler staple "$DMG"

# ── Sparkle Appcast ────────────────────────────────────────────────────
echo "=== Generating appcast ==="
if [ -z "$SPARKLE_TOOLS_DIR" ]; then
    echo "Warning: Sparkle sign_update tool not found — skipping appcast generation"
    echo "Run: find .build/artifacts -name sign_update"
else
    SIGN_OUTPUT=$("$SPARKLE_TOOLS_DIR/sign_update" "$DMG" 2>&1)
    ED_SIGNATURE=$(echo "$SIGN_OUTPUT" | sed -n 's/.*sparkle:edSignature="\([^"]*\)".*/\1/p')
    FILE_LENGTH=$(stat -f%z "$DMG")

    cat > "$RELEASE_DIR/appcast.xml" << APPCAST
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>Yuwp</title>
    <item>
      <title>Version $VERSION</title>
      <sparkle:version>$VERSION</sparkle:version>
      <sparkle:shortVersionString>$VERSION</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>14.0</sparkle:minimumSystemVersion>
      <enclosure url="https://github.com/duh17/yuwp/releases/download/v$VERSION/Yuwp-$VERSION.dmg"
                 type="application/octet-stream"
                 sparkle:edSignature="$ED_SIGNATURE"
                 length="$FILE_LENGTH" />
    </item>
  </channel>
</rss>
APPCAST
    echo "  appcast.xml generated"
fi

# ── Done ───────────────────────────────────────────────────────────────
echo ""
echo "=== Release $VERSION complete ==="
echo "Artifacts:"
echo "  $DMG"
[ -f "$RELEASE_DIR/appcast.xml" ] && echo "  $RELEASE_DIR/appcast.xml"
echo ""
echo "Upload to GitHub:"
echo "  gh release create v$VERSION '$DMG' '$RELEASE_DIR/appcast.xml' --title 'Yuwp $VERSION'"
