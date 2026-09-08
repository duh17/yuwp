#!/bin/bash
# Build a self-contained Yuwp.app, sign it, install to /Applications, then launch.
# The app bundle embeds:
#   - Yuwp
#   - yuwp-asr
#   - yuwp-tts
#   - mlx.metallib
#   - Sparkle.framework
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGURATION="release"
VERSION="${YUWP_VERSION:-0.1.0}"
: "${YUWP_INTERNAL_DIAGNOSTICS:=0}"
export YUWP_INTERNAL_DIAGNOSTICS
DEFAULT_SPARKLE_FEED_URL="https://github.com/duh17/yuwp/releases/latest/download/appcast.xml"
DEFAULT_SPARKLE_PUBLIC_ED_KEY="wnLCIfY048anOcj7/J/Iv6Lp9Fmba4zQ0EjCL7k/M+E=" # gitleaks:allow public Sparkle key
SPARKLE_FEED_URL="${YUWP_SPARKLE_FEED_URL:-$DEFAULT_SPARKLE_FEED_URL}"
SPARKLE_PUBLIC_ED_KEY="${YUWP_SPARKLE_PUBLIC_ED_KEY:-$DEFAULT_SPARKLE_PUBLIC_ED_KEY}"
APP="/Applications/Yuwp.app"
MACOS_DIR="$APP/Contents/MacOS"
RES_DIR="$APP/Contents/Resources"
FRAMEWORKS_DIR="$APP/Contents/Frameworks"
OPEN_SOURCE_DIR="$RES_DIR/OpenSource"
BIN_DIR=".build/arm64-apple-macosx/$CONFIGURATION"
VENDORED_LICENSES_DIR="third_party/licenses"
CAPTURE_RUN_LOG="${YUWP_CAPTURE_RUN_LOG:-0}"
LOGFILE="${YUWP_LOG_FILE:-/tmp/yuwp.log}"
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

# Remember a live TTS sidecar so we can replace it after the new binary is installed.
TTS_RESTART_CMD=""
TTS_PIDS=$(lsof -tiTCP:7937 -sTCP:LISTEN || true)
if [ -n "$TTS_PIDS" ]; then
    TTS_PID=$(echo "$TTS_PIDS" | awk 'NR==1 { print; exit }')
    TTS_CMD=$(ps -o command= -p "$TTS_PID" 2>/dev/null || true)
    case "$TTS_CMD" in
        *yuwp-tts*) TTS_RESTART_CMD=$TTS_CMD ;;
    esac
fi

# Stop the previous app and helpers so a restart cannot leave stale HTTP servers.
pkill -f "$APP/Contents/MacOS/Yuwp" >/dev/null 2>&1 || true
pkill -f "$APP/Contents/MacOS/yuwp-asr" >/dev/null 2>&1 || true
pkill -f "$APP/Contents/MacOS/yuwp-tts" >/dev/null 2>&1 || true
for port in 7936 7937; do
    OLD_SERVER_PIDS=$(lsof -tiTCP:$port -sTCP:LISTEN || true)
    if [ -n "$OLD_SERVER_PIDS" ]; then
        kill $OLD_SERVER_PIDS >/dev/null 2>&1 || true
    fi
done

# Rebuild replaces .build/release/yuwp-asr; bounce the LaunchAgent onto the new binary.
if launchctl print "gui/$(id -u)/com.yuwp.asr" >/dev/null 2>&1; then
    echo "[yuwp] Restarting LaunchAgent com.yuwp.asr"
    launchctl kickstart -k "gui/$(id -u)/com.yuwp.asr" >/dev/null 2>&1 || true
fi
sleep 1

# Update the app bundle in place to keep TCC identity stable across dev runs.
mkdir -p "$MACOS_DIR" "$RES_DIR" "$FRAMEWORKS_DIR"
rm -f "$MACOS_DIR/asr-server"
rm -f "$MACOS_DIR/yuwp-asr"
rm -f "$MACOS_DIR/yuwp-tts"
rm -f "$MACOS_DIR"/swift-*asr-server
rm -rf "$RES_DIR/Yuwp_NativeASR.bundle"
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

# Embed Sparkle.framework
SPARKLE_FW=".build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"
if [ -d "$SPARKLE_FW" ]; then
    rm -rf "$FRAMEWORKS_DIR/Sparkle.framework"
    ditto "$SPARKLE_FW" "$FRAMEWORKS_DIR/Sparkle.framework"
else
    echo "Warning: Sparkle.framework not found at $SPARKLE_FW — run 'swift package resolve' first"
fi

cat > "$APP/Contents/Info.plist" << EOF
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
EOF

# Code sign (inside-out: frameworks → helpers → main → bundle)
# Use hardened runtime when signing with a real identity
RUNTIME_FLAG=""
ENTITLEMENTS_APP=""
ENTITLEMENTS_SERVER=""
if [ "$SIGN_IDENTITY" != "-" ]; then
    RUNTIME_FLAG="--options runtime"
    ENTITLEMENTS_APP="--entitlements Yuwp.entitlements"
    ENTITLEMENTS_SERVER="--entitlements YuwpMLXHelper.entitlements"
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
    --identifier com.yuwp.app.asr "$MACOS_DIR/yuwp-asr"

codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG $ENTITLEMENTS_SERVER \
    --identifier com.yuwp.app.tts "$MACOS_DIR/yuwp-tts"

codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG $ENTITLEMENTS_APP \
    --identifier com.yuwp.app "$MACOS_DIR/Yuwp"

codesign --force --sign "$SIGN_IDENTITY" $RUNTIME_FLAG $ENTITLEMENTS_APP \
    --identifier com.yuwp.app "$APP"

codesign --verify --deep --strict "$APP"

if [ "$CAPTURE_RUN_LOG" = "1" ]; then
    > "$LOGFILE"
    echo "Launching Yuwp.app (log: $LOGFILE)..."
    open --stdout "$LOGFILE" --stderr "$LOGFILE" "$APP"
    sleep 4
    cat "$LOGFILE"
else
    echo "Launching Yuwp.app..."
    open "$APP"
fi

if [ -n "$TTS_RESTART_CMD" ]; then
    echo "[yuwp] Relaunching yuwp-tts sidecar on :7937"
    # Command was captured from ps of the previous listener.
    nohup $TTS_RESTART_CMD >>/tmp/yuwp-tts.log 2>&1 &
    disown || true
fi
