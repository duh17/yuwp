#!/bin/bash
# Build, sign, notarize, and package a Yuwp release.
#
# Usage: release.sh <version>
#   e.g.: release.sh 0.2.0
#
# Required environment variables:
#   YUWP_SIGN_IDENTITY  — Developer ID Application identity
#   Notarization auth (choose one):
#     - YUWP_NOTARY_PROFILE (preferred; keychain profile for notarytool)
#     - YUWP_TEAM_ID + YUWP_APPLE_ID + YUWP_APP_PASSWORD
#   YUWP_SPARKLE_FEED_URL   — optional Sparkle appcast URL override
#   YUWP_SPARKLE_PUBLIC_ED_KEY — optional Sparkle public key override
#
# Output:
#   release/Yuwp-<version>.dmg  — notarized, stapled disk image
#   release/appcast.xml         — Sparkle appcast entry
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION="${1:?Usage: release.sh <version>}"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must use numeric SemVer form (for example, 0.1.3)"
    exit 1
fi
SIGN_IDENTITY="${YUWP_SIGN_IDENTITY:?Set YUWP_SIGN_IDENTITY}"
NOTARY_PROFILE="${YUWP_NOTARY_PROFILE:-}"
TEAM_ID="${YUWP_TEAM_ID:-}"
APPLE_ID="${YUWP_APPLE_ID:-}"
APP_PASSWORD="${YUWP_APP_PASSWORD:-}"

if [ -z "$NOTARY_PROFILE" ]; then
    [ -n "$TEAM_ID" ] || { echo "Set YUWP_NOTARY_PROFILE, or set YUWP_TEAM_ID + YUWP_APPLE_ID + YUWP_APP_PASSWORD"; exit 1; }
    [ -n "$APPLE_ID" ] || { echo "Set YUWP_NOTARY_PROFILE, or set YUWP_TEAM_ID + YUWP_APPLE_ID + YUWP_APP_PASSWORD"; exit 1; }
    [ -n "$APP_PASSWORD" ] || { echo "Set YUWP_NOTARY_PROFILE, or set YUWP_TEAM_ID + YUWP_APPLE_ID + YUWP_APP_PASSWORD"; exit 1; }
fi

DEFAULT_SPARKLE_FEED_URL="https://github.com/duh17/yuwp/releases/latest/download/appcast.xml"
DEFAULT_SPARKLE_PUBLIC_ED_KEY="wnLCIfY048anOcj7/J/Iv6Lp9Fmba4zQ0EjCL7k/M+E=" # gitleaks:allow public Sparkle key
SPARKLE_FEED_URL="${YUWP_SPARKLE_FEED_URL:-$DEFAULT_SPARKLE_FEED_URL}"
SPARKLE_PUBLIC_ED_KEY="${YUWP_SPARKLE_PUBLIC_ED_KEY:-$DEFAULT_SPARKLE_PUBLIC_ED_KEY}"

CONFIGURATION="release"
export YUWP_INTERNAL_DIAGNOSTICS=0
RELEASE_DIR="release"
APP="$RELEASE_DIR/Yuwp.app"
MACOS_DIR="$APP/Contents/MacOS"
RES_DIR="$APP/Contents/Resources"
FRAMEWORKS_DIR="$APP/Contents/Frameworks"
OPEN_SOURCE_DIR="$RES_DIR/OpenSource"
DMG_STAGE="$RELEASE_DIR/dmg-root"
DMG="$RELEASE_DIR/Yuwp-$VERSION.dmg"
VENDORED_LICENSES_DIR="third_party/licenses"

SPARKLE_FW=".build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"
SPARKLE_TOOLS_DIR=".build/artifacts/sparkle/Sparkle/bin"
SPARKLE_SIGN_UPDATE="$SPARKLE_TOOLS_DIR/sign_update"
SPARKLE_GENERATE_KEYS="$SPARKLE_TOOLS_DIR/generate_keys"

[ -x "$SPARKLE_SIGN_UPDATE" ] || { echo "Error: missing Sparkle sign_update at $SPARKLE_SIGN_UPDATE"; exit 1; }
[ -x "$SPARKLE_GENERATE_KEYS" ] || { echo "Error: missing Sparkle generate_keys at $SPARKLE_GENERATE_KEYS"; exit 1; }
KEYCHAIN_PUBLIC_ED_KEY=$("$SPARKLE_GENERATE_KEYS" -p)
if [ "$KEYCHAIN_PUBLIC_ED_KEY" != "$SPARKLE_PUBLIC_ED_KEY" ]; then
    echo "Error: configured Sparkle public key does not match the private key in Keychain"
    exit 1
fi

# ── Build ──────────────────────────────────────────────────────────────
echo "=== Building Yuwp $VERSION ==="
bash scripts/build.sh "$CONFIGURATION"
BIN_DIR=$(swift build --show-bin-path -c "$CONFIGURATION")

# ── Assemble Bundle ────────────────────────────────────────────────────
echo "=== Assembling app bundle ==="
rm -rf "$RELEASE_DIR"
mkdir -p "$MACOS_DIR" "$RES_DIR" "$FRAMEWORKS_DIR"

cp -f "$BIN_DIR/Yuwp" "$MACOS_DIR/Yuwp"
cp -f "$BIN_DIR/yuwp-asr" "$MACOS_DIR/yuwp-asr"
cp -f "$BIN_DIR/yuwp-tts" "$MACOS_DIR/yuwp-tts"
cp -f "$BIN_DIR/mlx.metallib" "$MACOS_DIR/mlx.metallib"
cp -f "Resources/Yuwp.icns" "$RES_DIR/Yuwp.icns"

RESOURCE_BUNDLE="$BIN_DIR/Yuwp_NativeASR.bundle"
if [ -d "$RESOURCE_BUNDLE" ]; then
    ditto "$RESOURCE_BUNDLE" "$RES_DIR/Yuwp_NativeASR.bundle"
else
    echo "Error: missing NativeASR resource bundle at $RESOURCE_BUNDLE"
    exit 1
fi

if [ ! -d "$VENDORED_LICENSES_DIR" ]; then
    echo "Error: missing vendored licenses at $VENDORED_LICENSES_DIR"
    exit 1
fi
rm -rf "$OPEN_SOURCE_DIR"
mkdir -p "$OPEN_SOURCE_DIR"
cp -f "LICENSE" "$OPEN_SOURCE_DIR/LICENSE.txt"
cp -f "THIRD_PARTY_NOTICES.md" "$OPEN_SOURCE_DIR/THIRD_PARTY_NOTICES.md"
ditto "$VENDORED_LICENSES_DIR" "$OPEN_SOURCE_DIR/licenses"

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
    <key>SUEnableAutomaticChecks</key>
    <true/>
    <key>SUAutomaticallyUpdate</key>
    <true/>
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

# yuwp-asr (needs allow-unsigned-executable-memory for MLX)
codesign --force --sign "$SIGN_IDENTITY" --options runtime \
    --entitlements YuwpMLXHelper.entitlements \
    --identifier com.yuwp.app.asr "$MACOS_DIR/yuwp-asr"

# yuwp-tts (needs allow-unsigned-executable-memory for MLX)
codesign --force --sign "$SIGN_IDENTITY" --options runtime \
    --entitlements YuwpMLXHelper.entitlements \
    --identifier com.yuwp.app.tts "$MACOS_DIR/yuwp-tts"

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
mkdir -p "$DMG_STAGE"
ditto "$APP" "$DMG_STAGE/Yuwp.app"
cp -f "LICENSE" "$DMG_STAGE/LICENSE.txt"
cp -f "THIRD_PARTY_NOTICES.md" "$DMG_STAGE/THIRD_PARTY_NOTICES.md"
ditto "$VENDORED_LICENSES_DIR" "$DMG_STAGE/THIRD_PARTY_LICENSES"
hdiutil create -volname "Yuwp" -srcfolder "$DMG_STAGE" \
    -ov -format UDZO "$DMG"
rm -rf "$DMG_STAGE"

codesign --force --sign "$SIGN_IDENTITY" "$DMG"

# ── Notarize ───────────────────────────────────────────────────────────
echo "=== Notarizing (this may take several minutes) ==="
if [ -n "$NOTARY_PROFILE" ]; then
    echo "Using notarytool keychain profile: $NOTARY_PROFILE"
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
SIGN_OUTPUT=$("$SPARKLE_SIGN_UPDATE" "$DMG" 2>&1)
ED_SIGNATURE=$(echo "$SIGN_OUTPUT" | sed -n 's/.*sparkle:edSignature="\([^"]*\)".*/\1/p')
[ -n "$ED_SIGNATURE" ] || { echo "Error: Sparkle sign_update returned no EdDSA signature"; exit 1; }
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
xmllint --noout "$RELEASE_DIR/appcast.xml"
grep -Fq "sparkle:edSignature=\"$ED_SIGNATURE\"" "$RELEASE_DIR/appcast.xml"
grep -Fq "length=\"$FILE_LENGTH\"" "$RELEASE_DIR/appcast.xml"
echo "  appcast.xml generated and validated"

# ── Done ───────────────────────────────────────────────────────────────
echo ""
echo "=== Release $VERSION complete ==="
echo "Artifacts:"
echo "  $DMG"
[ -f "$RELEASE_DIR/appcast.xml" ] && echo "  $RELEASE_DIR/appcast.xml"
echo ""
echo "Upload to GitHub:"
echo "  gh release create v$VERSION '$DMG' '$RELEASE_DIR/appcast.xml' --title 'Yuwp $VERSION'"
