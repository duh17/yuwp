#!/bin/bash
# Build and install Yuwp to /Applications, then launch.
# Installing to /Applications with Developer ID signing gives stable
# TCC permissions (Accessibility + Microphone) across rebuilds.
set -e

cd "$(dirname "$0")/.."

echo "Building..."
swift build 2>&1 | tail -3

# Install to /Applications
APP="/Applications/Yuwp.app"
mkdir -p "$APP/Contents/MacOS"
mkdir -p "$APP/Contents/Resources"
cp .build/debug/Yuwp "$APP/Contents/MacOS/"

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

# Sign with Developer ID for stable TCC permissions
codesign --force --sign "Developer ID Application: Da Chen (AZAQMY4SPZ)" \
    --identifier com.yuwp.app "$APP/Contents/MacOS/Yuwp" 2>/dev/null
codesign --force --sign "Developer ID Application: Da Chen (AZAQMY4SPZ)" \
    --identifier com.yuwp.app "$APP" 2>/dev/null

LOGFILE="/tmp/yuwp.log"
> "$LOGFILE"

echo "Launching Yuwp.app (log: $LOGFILE)..."
open --stdout "$LOGFILE" --stderr "$LOGFILE" "$APP"
sleep 3
cat "$LOGFILE"
