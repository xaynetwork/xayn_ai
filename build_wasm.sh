#!/bin/bash

set -e

build_wasm() {
    local WASM_OUT_DIR=$1
    local WASM_FEATURE=$2
    local CARGO_ARGS=${@:3}

    local PACKAGE=$WASM_OUT_DIR/package.json

    wasm-pack build xayn-ai-ffi-wasm \
        --no-typescript \
        --out-dir ../$WASM_OUT_DIR \
        --out-name genesis \
        --target web \
        --release \
        -- $CARGO_ARGS
    # remove glob gitignore (https://rustwasm.github.io/docs/wasm-pack/commands/build.html#footnote-0)
    rm $WASM_OUT_DIR/.gitignore

    local TMP_FILE=$(mktemp)
    jq --arg feature $WASM_FEATURE '. += {"feature": $feature}' "$PACKAGE" > $TMP_FILE
    mv $TMP_FILE $PACKAGE
}

build_wasm "$1" "$2" ${@:3}
