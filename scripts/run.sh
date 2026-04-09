#!/bin/bash
# Build a self-contained Yuwp.app, sign it, install to /Applications, then launch.
# The app bundle embeds:
#   - Yuwp
#   - asr-server
#   - mlx.metallib
#   - Sparkle.framework
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGURATION="release"
: "${YUWP_INTERNAL_DIAGNOSTICS:=1}"
APP="/Applications/Yuwp.app"
MACOS_DIR="$APP/Contents/MacOS"
RES_DIR="$APP/Contents/Resources"
FRAMEWORKS_DIR="$APP/Contents/Frameworks"
BIN_DIR=".build/arm64-apple-macosx/$CONFIGURATION"
LOGFILE="/tmp/yuwp.log"
SIGN_IDENTITY="${YUWP_SIGN_IDENTITY:-}"
if [ -z "$SIGN_IDENTITY" ]; then
    SIGN_IDENTITY=$(security find-identity -v -p codesigning 2>/dev/null \
        | awk -F '"' '/Developer ID Application:/ { print $2 }' \
        | awk '!seen[$0]++' \
        | head -n 1)
    if [ -n "$SIGN_IDENTITY" ]; then
        echo "[yuwp] Using auto-detected signing identity: $SIGN_IDENTITY"
    else
        SIGN_IDENTITY="-"
        echo "[yuwp] Warning: no Developer ID Application identity found; falling back to ad-hoc signing"
    fi
elif [ "$SIGN_IDENTITY" = "-" ]; then
    echo "[yuwp] Using ad-hoc signing (YUWP_SIGN_IDENTITY=-)"
else
    echo "[yuwp] Using configured signing identity: $SIGN_IDENTITY"
fi

echo "[yuwp] Internal diagnostics env: $YUWP_INTERNAL_DIAGNOSTICS"
bash scripts/build.sh "$CONFIGURATION"

# Kill old app binary if it is already running so we don't leave a stale copy alive.
pkill -f "$APP/Contents/MacOS/Yuwp" >/dev/null 2>&1 || true

# Clear any stale ASR server still holding the port from an old dev run.
OLD_SERVER_PIDS=$(lsof -tiTCP:9748 -sTCP:LISTEN || true)
if [ -n "$OLD_SERVER_PIDS" ]; then
    kill $OLD_SERVER_PIDS >/dev/null 2>&1 || true
fi
sleep 1

# Recreate the bundle from scratch so embedded frameworks/helpers don't accumulate
# stale contents across repeated runs.
rm -rf "$APP"
mkdir -p "$MACOS_DIR" "$RES_DIR" "$FRAMEWORKS_DIR"
cp -f "$BIN_DIR/Yuwp" "$MACOS_DIR/Yuwp"
cp -f "$BIN_DIR/asr-server" "$MACOS_DIR/asr-server"
cp -f "$BIN_DIR/mlx.metallib" "$MACOS_DIR/mlx.metallib"
cp -f "icon-layers/Yuwp.icns" "$RES_DIR/Yuwp.icns"

# SwiftPM doesn't add the app-bundle Frameworks runpath for this executable.
# Add it here so the packaged app can load Sparkle.framework at runtime.
install_name_tool -add_rpath "@executable_path/../Frameworks" "$MACOS_DIR/Yuwp"

# Embed Sparkle.framework
SPARKLE_FW=".build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"
if [ -d "$SPARKLE_FW" ]; then
    ditto "$SPARKLE_FW" "$FRAMEWORKS_DIR/Sparkle.framework"
else
    echo "Warning: Sparkle.framework not found at $SPARKLE_FW — run 'swift package resolve' first"
fi

cat > "$APP/Contents/Info.plist" << 'EOF'
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
    <string>1</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>Yuwp</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Yuwp needs microphone access to transcribe your speech into text.</string>
    <key>SUFeedURL</key>
    <string>https://github.com/duh17/yuwp/releases/latest/download/appcast.xml</string>
    <key>SUPublicEDKey</key>
    <string></string>
</dict>
</plist>
EOF

# Code sign (inside-out: frameworks → helpers → main → bundle)
# Use hardened runtime when signing with a real identity
RUNTIME_FLAG=""
ENTITLEMENTS_APP=""
ENTITLEMENTS_SERVER=""
if [ "$SIGN_IDENTITY" != "-" ]; then
    RUNTIME_FLAG="--options runtime"
    ENTITLEMENTS_APP="--entitlements Yuwp.entitlements"
    ENTITLEMENTS_SERVER="--entitlements asr-server.entitlements"
fi

# Sparkle framework (XPC services, helpers, then the framework itself)
if [ -d "$FRAMEWORKS_DIR/Sparkle.framework" ]; then
    SPARKLE_VERSION_DIR="$FRAMEWORKS_DIR/Sparkle.framework/Versions/B"
    for xpc in "$SPARKLE_VERSION_DIR"/XPCServices/*.xpc; do
        [ -e "$xpc" ] && codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG "$xpc"
    done
    if [ -d "$SPARKLE_VERSION_DIR/Updater.app" ]; then
        codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG "$SPARKLE_VERSION_DIR/Updater.app"
    fi
    if [ -f "$SPARKLE_VERSION_DIR/Autoupdate" ]; then
        codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG "$SPARKLE_VERSION_DIR/Autoupdate"
    fi
    codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG "$FRAMEWORKS_DIR/Sparkle.framework"
fi

codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG \
    --identifier com.yuwp.app.metallib "$MACOS_DIR/mlx.metallib"

codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG $ENTITLEMENTS_SERVER \
    --identifier com.yuwp.app.server "$MACOS_DIR/asr-server"

codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG $ENTITLEMENTS_APP \
    --identifier com.yuwp.app "$MACOS_DIR/Yuwp"

codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG $ENTITLEMENTS_APP \
    --identifier com.yuwp.app "$APP"

codesign --verify --deep --strict "$APP"

> "$LOGFILE"
echo "Launching Yuwp.app (log: $LOGFILE)..."
open --stdout "$LOGFILE" --stderr "$LOGFILE" "$APP"
sleep 4
cat "$LOGFILE"
