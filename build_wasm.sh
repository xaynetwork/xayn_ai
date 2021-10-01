#!/bin/bash

# Builds `xayn-ai-ffi-wasm` and adds the given feature to the generated `package.json`.
#
# The script needs to be executed in the root of the repository.
#
# Usage:
# ./build_wasm <absolute path of the wasm output directory> <name of the feature e.g. sequential> [<additional cargo args>]

set -e

build_wasm() {
    local WASM_OUT_DIR_PATH=$1
    local WASM_FEATURE=$2
    local CARGO_ARGS=${@:3}

    echo "build $WASM_FEATURE wasm"

    wasm-pack build xayn-ai-ffi-wasm \
        --no-typescript \
        --out-dir "$WASM_OUT_DIR_PATH" \
        --out-name genesis \
        --target web \
        --release \
        -- $CARGO_ARGS
    # remove glob gitignore (https://rustwasm.github.io/docs/wasm-pack/commands/build.html#footnote-0)
    rm "$WASM_OUT_DIR_PATH/.gitignore"

    local TMP_FILE=$(mktemp)
    local PACKAGE="$WASM_OUT_DIR_PATH/package.json"
    jq --arg feature "$WASM_FEATURE" '. += {"feature": $feature}' "$PACKAGE" > "$TMP_FILE"
    mv "$TMP_FILE" "$PACKAGE"
}

build_wasm "$1" "$2" ${@:3}
