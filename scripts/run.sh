#!/bin/bash
# Build and run Yuwp as a proper .app bundle.
# The app bundle is needed so macOS TCC can identify the binary
# and properly prompt for Microphone + Accessibility permissions.
set -e

cd "$(dirname "$0")/.."

echo "Building..."
swift build 2>&1 | tail -3

# Create .app bundle
APP=".build/Yuwp.app"
mkdir -p "$APP/Contents/MacOS"
mkdir -p "$APP/Contents/Resources"
cp .build/debug/Yuwp "$APP/Contents/MacOS/"

# Sign with stable identifier so TCC permissions persist across rebuilds
codesign --force --sign - --identifier com.yuwp.app "$APP/Contents/MacOS/Yuwp" 2>/dev/null
codesign --force --sign - --identifier com.yuwp.app "$APP" 2>/dev/null

# Info.plist with required permission descriptions
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

LOGFILE="/tmp/yuwp.log"
> "$LOGFILE"

echo "Launching Yuwp.app (log: $LOGFILE)..."
open --stdout "$LOGFILE" --stderr "$LOGFILE" "$APP"
sleep 3
cat "$LOGFILE"
