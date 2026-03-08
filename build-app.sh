#!/bin/bash
# Build PTT.app for distribution
# Creates a self-contained .app bundle with ptt.py copied in
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/dist"
APP="$BUILD_DIR/PTT.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# Copy Info.plist
cp "$SCRIPT_DIR/PTT.app/Contents/Info.plist" "$APP/Contents/Info.plist"

# Copy launcher
cp "$SCRIPT_DIR/PTT.app/Contents/MacOS/ptt-launcher" "$APP/Contents/MacOS/ptt-launcher"
chmod +x "$APP/Contents/MacOS/ptt-launcher"

# Copy ptt.py (actual copy, not symlink)
cp "$SCRIPT_DIR/ptt.py" "$APP/Contents/Resources/ptt.py"

echo ""
echo "  Built: $APP"
echo ""
echo "  To install:"
echo "    cp -r dist/PTT.app /Applications/"
echo ""
echo "  Note: Requires uv (brew install uv) and Accessibility permission."
echo ""
