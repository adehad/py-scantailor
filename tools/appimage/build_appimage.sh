#!/usr/bin/env bash
# Package the PyInstaller folder build as an AppImage.
#
# Usage: tools/appimage/build_appimage.sh dist/ScanTailor ScanTailor-x86_64.AppImage
set -euo pipefail

APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"
APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
RUNTIME_URL="https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64"
RUNTIME_SHA256="2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d"

bundle="$(realpath "$1")"
output="$(realpath -m "$2")"
here="$(dirname "$(realpath "$0")")"
resources="$here/../../src/scantailor/resources"
app_id="io.github.adehad.py-scantailor"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fetch() {
  curl -sSfL --retry 3 -o "$work/$1" "$2"
  echo "$3  $work/$1" | sha256sum -c -
}
fetch appimagetool "$APPIMAGETOOL_URL" "$APPIMAGETOOL_SHA256"
fetch runtime "$RUNTIME_URL" "$RUNTIME_SHA256"
chmod +x "$work/appimagetool"

appdir="$work/AppDir"
mkdir -p "$appdir/usr/lib" "$appdir/usr/share/applications" "$appdir/usr/share/metainfo" \
  "$appdir/usr/share/icons/hicolor/256x256/apps" "$appdir/usr/share/icons/hicolor/scalable/apps" \
  "$appdir/usr/share/mime/packages"
cp -a "$bundle" "$appdir/usr/lib/ScanTailor"
install -m 755 "$here/AppRun" "$appdir/AppRun"
install -m 644 "$here/$app_id.desktop" "$appdir/$app_id.desktop"
install -m 644 "$here/$app_id.desktop" "$appdir/usr/share/applications/$app_id.desktop"
install -m 644 "$resources/unix/mime/scantailor-project.xml" "$appdir/usr/share/mime/packages/$app_id.xml"
install -m 644 "$here/$app_id.appdata.xml" "$appdir/usr/share/metainfo/$app_id.appdata.xml"
install -m 644 "$resources/ScanTailor.png" "$appdir/scantailor.png"
install -m 644 "$resources/ScanTailor.png" "$appdir/usr/share/icons/hicolor/256x256/apps/scantailor.png"
install -m 644 "$resources/appicon.svg" "$appdir/usr/share/icons/hicolor/scalable/apps/scantailor.svg"
ln -s scantailor.png "$appdir/.DirIcon"

# appimagetool is itself an AppImage, and CI runners have no FUSE to mount it.
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$work/appimagetool" --no-appstream \
  --runtime-file "$work/runtime" "$appdir" "$output"
