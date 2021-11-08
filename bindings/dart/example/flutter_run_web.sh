#!/bin/sh

set -eu

# This provides a way to run flutter web which is compatible with using
# threads in wasm. In difference to e.g. `flutter run -d Chrome` this doesn't
# support live reloading and similar features. The reason for this is that
# we need to run a custom server to set the required http headers.

error() {
    echo $1 >&2
    exit 1
}

ROOT="$(dirname $0)"
cd "$ROOT"
BUILD_OUT="./build/web"
CANVASKIT_OUT="$BUILD_OUT/canvaskit"

# The default canvaskit is hosted on `https://unpkg.com/` but that CDN
# doesn't yet set the Cross-Origin-Resource-Policy header. This makes it
# unusable if our site uses `Cross-Origin-Embedder-Policy: require-corp`.
# But we need to set that header to be able to use `SharedArrayBuffer`.
#
# Issue: https://github.com/mjackson/unpkg/issues/290
flutter build web \
    --dart-define=FLUTTER_WEB_CANVASKIT_URL=./canvaskit/

# Fetch JS libs from CDN to avoid problems with COOP/COEP

# Usage: download_unpkg <on-cdn-project> <version> <file-name> [(<output_dir>|"") [<on-cdn-in-dir>]]
#
# Downloads a file from the unpkg CDN.
#
# Output folders will be created like necessary.
download_unpkg() {
    if [ -z "$4" ]; then
        OUT="$3"
    else
        OUT="$4/$3"
    fi
    mkdir -p "$(dirname "$OUT")"
    curl "https://unpkg.com/$1@$2${5:-}/$3" > "$OUT"
}

# Downloads a file from the canvaskit project which was placed in the `bin/` directory.
download_canvaskit_file() {
    OUT="$CANVASKIT_OUT/$1"
    download_unpkg canvaskit-wasm 0.28.1 $1 $CANVASKIT_OUT /bin
}
if [ ! -e "$CANVASKIT_OUT/canvaskit.js" ]; then
    echo "Downloading canvaskit" >&2
    mkdir -p "$CANVASKIT_OUT"
    download_canvaskit_file "canvaskit.js"
    download_canvaskit_file "canvaskit.wasm"
    download_canvaskit_file "profiling/canvaskit.js"
    download_canvaskit_file "profiling/canvaskit.wasm"
fi

echo "Running Server"
/usr/bin/env python3 "./server.py" "$BUILD_OUT"
