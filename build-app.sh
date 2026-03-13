#!/bin/bash
# Build PTT.app and PTT.dmg for distribution
# Creates a drag-to-install disk image
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/dist"
APP="$BUILD_DIR/PTT.app"
DMG="$BUILD_DIR/PTT.dmg"
DMG_STAGING="$BUILD_DIR/dmg-staging"

# Clean previous builds
rm -rf "$APP" "$DMG" "$DMG_STAGING"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# Copy Info.plist
cp "$SCRIPT_DIR/PTT.app/Contents/Info.plist" "$APP/Contents/Info.plist"

# Copy launcher
cp "$SCRIPT_DIR/PTT.app/Contents/MacOS/ptt-launcher" "$APP/Contents/MacOS/ptt-launcher"
chmod +x "$APP/Contents/MacOS/ptt-launcher"

# Copy ptt.py
cp "$SCRIPT_DIR/ptt.py" "$APP/Contents/Resources/ptt.py"

# Build DMG
echo ""
echo "  Building DMG..."

mkdir -p "$DMG_STAGING"
cp -R "$APP" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"

hdiutil create -volname "PTT" \
    -srcfolder "$DMG_STAGING" \
    -ov -format UDZO \
    "$DMG" \
    -quiet

rm -rf "$DMG_STAGING"

echo ""
echo "  Built:"
echo "    $APP"
echo "    $DMG"
echo ""
echo "  Share the DMG — your friend opens it and drags PTT to Applications."
echo "  First launch downloads the AI model (~460 MB) and asks for Accessibility permission."
echo ""
