#!/bin/bash
# Build a self-contained Yuwp.app, sign it, install to /Applications, then launch.
# The app bundle embeds:
#   - Yuwp
#   - asr-server
#   - mlx.metallib
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGURATION="release"
APP="/Applications/Yuwp.app"
MACOS_DIR="$APP/Contents/MacOS"
RES_DIR="$APP/Contents/Resources"
BIN_DIR=".build/arm64-apple-macosx/$CONFIGURATION"
LOGFILE="/tmp/yuwp.log"
SIGN_IDENTITY="Developer ID Application: Da Chen (AZAQMY4SPZ)"

bash scripts/build.sh "$CONFIGURATION"

mkdir -p "$MACOS_DIR" "$RES_DIR"
cp -f "$BIN_DIR/Yuwp" "$MACOS_DIR/Yuwp"
cp -f "$BIN_DIR/asr-server" "$MACOS_DIR/asr-server"
cp -f "$BIN_DIR/mlx.metallib" "$MACOS_DIR/mlx.metallib"

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
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Yuwp needs microphone access to transcribe your speech into text.</string>
</dict>
</plist>
EOF

# Kill old app binary if it is already running so we don't leave a stale copy alive.
pkill -f "$APP/Contents/MacOS/Yuwp" >/dev/null 2>&1 || true

# Clear any stale ASR server still holding the port from an old dev run.
OLD_SERVER_PIDS=$(lsof -tiTCP:9748 -sTCP:LISTEN || true)
if [ -n "$OLD_SERVER_PIDS" ]; then
    kill $OLD_SERVER_PIDS >/dev/null 2>&1 || true
fi
sleep 1

codesign --force --sign "$SIGN_IDENTITY" --identifier com.yuwp.app.metallib "$MACOS_DIR/mlx.metallib"
codesign --force --sign "$SIGN_IDENTITY" --identifier com.yuwp.app.server "$MACOS_DIR/asr-server"
codesign --force --sign "$SIGN_IDENTITY" --identifier com.yuwp.app "$MACOS_DIR/Yuwp"
codesign --force --sign "$SIGN_IDENTITY" --identifier com.yuwp.app "$APP"

> "$LOGFILE"
echo "Launching Yuwp.app (log: $LOGFILE)..."
open --stdout "$LOGFILE" --stderr "$LOGFILE" "$APP"
sleep 4
cat "$LOGFILE"
