#!/bin/bash

# Builds `xayn-ai-ffi-wasm` and adds the given feature to the generated `package.json`.
#
# The script needs to be executed in the root of the repository.
# The name of the feature can consists of multiple words separated by whitespaces.
# For example: "multithreaded simd".
#
# Usage:
# ./build_wasm <absolute path of the wasm output directory> <name of the feature e.g. sequential> [<additional cargo args>]

set -e

build_wasm() {
    local WASM_OUT_DIR_PATH=$1
    local WASM_FEATURE=$2
    local CARGO_ARGS=${@:3}

    echo "build $WASM_FEATURE wasm"

    local TMP_DIR=$(mktemp -d)
    wasm-pack build xayn-ai-ffi-wasm \
        --no-typescript \
        --out-dir "$TMP_DIR" \
        --out-name genesis \
        --target web \
        --release \
        -- $CARGO_ARGS

    local BUNDLER_CONFIG="data/bundler_config/webpack.config.js"
    webpack "$TMP_DIR/genesis.js" -o "$WASM_OUT_DIR_PATH" -c "$BUNDLER_CONFIG"
    mv "$TMP_DIR/package.json" "$WASM_OUT_DIR_PATH"
    rm -r "$TMP_DIR"

    local TMP_FILE=$(mktemp)
    local PACKAGE="$WASM_OUT_DIR_PATH/package.json"
    jq --arg feature "$WASM_FEATURE" '. += {"feature": $feature}' "$PACKAGE" > "$TMP_FILE"
    mv "$TMP_FILE" "$PACKAGE"
}

build_wasm "$1" "$2" ${@:3}
