#!/bin/bash

# Generates the metadata of the assets (ai and wasm). If an asset of the `data_assets` array
# contains a `chunk_size` key, the script splits the asset into chunks where each chunk has
# a maximum size of `chunk_size`. The format of the `chunk_size` value is equivalent to the
# `SIZE` argument in `split` or `gsplit` on macOS. See `split`/`gsplit` man page for more details.
#
# The script needs to be executed in the root of the repository.
#
# Usage:
# ./generate_assets_metadata <path of assets_manifest.json> [<path of the wasm output directory>] [<wasm sequential features version>] [<wasm parallel feature version>]

set -e

OUT_DIR="out"
ASSETS_METADATA_PATH=$OUT_DIR/assets_metadata.json

if [[ "$OSTYPE" == "darwin"*  || $RUNNER_OS == "macOS" ]]; then
        if [ -x "$(command -v gsplit)" ]; then
            SPLIT="gsplit"
        else
            echo "Requires the GNU version of 'split'. Use 'brew install coreutils' to install it."
            exit 1
        fi
else
        SPLIT="split"
fi

if ! [ -x "$(command -v gomplate)" ]; then
    echo "Cannot find 'gomplate'."
    exit 1
fi

if ! [ -x "$(command -v jq)" ]; then
    echo "Cannot find 'jq'."
    exit 1
fi

split_asset_into_chunks(){
    local CHUNKS_DIR=$1
    local ASSET_CHUNK_SIZE=$2
    local ASSET_PATH=$3
    local ASSET_FILENAME=$4
    local ASSET_VERSION=$5

    mkdir ${CHUNKS_DIR}/${ASSET_VERSION}

    $SPLIT --numeric-suffixes=0 -b $ASSET_CHUNK_SIZE $ASSET_PATH "${CHUNKS_DIR}/${ASSET_VERSION}/${ASSET_FILENAME}_${ASSET_CHUNK_SIZE}_"
}

generate_dart_assets_manifest() {
    gomplate -d assets_manifest=$ASSETS_METADATA_PATH -f data/asset_templates/assets.dart.tmpl -o bindings/dart/lib/src/common/reranker/assets.dart
    flutter format bindings/dart/lib/src/common/reranker/assets.dart
}

calc_checksum() {
    echo $(shasum -a 256 $1 | awk '{ print $1 }')
}

gen_data_assets_metadata() {
    local ASSET_MANIFEST=$1
    local DATA_DIR="data"
    local CHUNKS_DIR=$DATA_DIR/chunks
    local TMP_FILE=$(mktemp)

    $(rm -rf $CHUNKS_DIR || true) && mkdir $CHUNKS_DIR

    for ASSET in $(cat $ASSET_MANIFEST | jq -c '.data_assets[]'); do
        local ASSET_URL_SUFFIX=$(echo $ASSET | jq -r '.url_suffix')
        local ASSET_PATH="$DATA_DIR/$ASSET_URL_SUFFIX"
        local ASSET_CHECKSUM=$(calc_checksum $ASSET_PATH)
        local ASSET_WITH_CHECKSUM=$(echo $ASSET | jq -c --arg checksum $ASSET_CHECKSUM --arg path $ASSET_PATH '. |= .+ {"checksum": $checksum, "path": $path}')

        local UPDATED_ASSET=$ASSET_WITH_CHECKSUM
        local ASSET_CHUNK_SIZE=$(echo $ASSET | jq -r '.chunk_size')
        if [ "$ASSET_CHUNK_SIZE" != "null" ]; then
            local ASSET_FILENAME=$(basename $ASSET_URL_SUFFIX)
            local ASSET_VERSION=$(dirname $ASSET_URL_SUFFIX)

            split_asset_into_chunks $CHUNKS_DIR $ASSET_CHUNK_SIZE $ASSET_PATH $ASSET_FILENAME $ASSET_VERSION

            local FRAGMENTS=""
            for CHUNK_PATH in $(find ${CHUNKS_DIR}/${ASSET_VERSION} -name "${ASSET_FILENAME}_*" | sort -n); do
                local FRAGMENT_CHECKSUM=$(calc_checksum $CHUNK_PATH)
                local FRAGMENT_FILENAME=$(basename $CHUNK_PATH)
                local FRAGMENT="{\"path\": \"$CHUNK_PATH\", \"url_suffix\": \"${ASSET_VERSION}/${FRAGMENT_FILENAME}\", \"checksum\": \"$FRAGMENT_CHECKSUM\"}"
                if [ -z "${FRAGMENTS}" ]; then
                    FRAGMENTS="$FRAGMENT"
                else
                    FRAGMENTS+=", $FRAGMENT"
                fi
            done

            local UPDATED_ASSET=$(echo $ASSET_WITH_CHECKSUM | jq -c --argjson fragments "[$FRAGMENTS]" '. |= .+ {"fragments": $fragments}' | jq -c 'del(.chunk_size)')
        else
            local UPDATED_ASSET=$(echo $ASSET_WITH_CHECKSUM | jq -c '. |= .+ {"fragments": []}')
        fi

        jq --argjson asset $UPDATED_ASSET '.assets |= .+ [$asset]' $ASSETS_METADATA_PATH > $TMP_FILE
        mv $TMP_FILE $ASSETS_METADATA_PATH
    done
}

gen_wasm_asset_metadata() {
    local DART_ENUM_NAME=$1
    local WASM_VERSION=$2
    local ASSET_FILENAME=$3
    local ASSET_PATH=$4

    local TMP_FILE=$(mktemp)

    local ASSET="{\"dart_enum_name\": \"$DART_ENUM_NAME\", \"fragments\": []}"

    if [ -f "$ASSET_PATH" ]; then
        local ASSET_CHECKSUM=$(calc_checksum $ASSET_PATH)
        local ASSET_WITH_CHECKSUM=$(echo $ASSET | jq -c --arg checksum $ASSET_CHECKSUM '. |= .+ {"checksum": $checksum}')
        local ASSET_URL_SUFFIX=${WASM_VERSION}/${ASSET_FILENAME}
        ASSET=$(echo $ASSET_WITH_CHECKSUM | jq -c --arg path $ASSET_PATH --arg url_suffix $ASSET_URL_SUFFIX '. |= .+ {"path": $path, "url_suffix": $url_suffix}')
    else
        ASSET=$(echo $ASSET | jq -c '. |= .+ {"url_suffix": "", "checksum": ""}')
    fi

    jq --argjson wasm_asset $ASSET '.assets |= .+ [$wasm_asset]' $ASSETS_METADATA_PATH > $TMP_FILE
    mv $TMP_FILE $ASSETS_METADATA_PATH
}

gen_wasm_assets_metadata() {
    local WASM_FEATURE=$1
    local WASM_VERSION=$2
    local WASM_OUT_DIR_PATH=$3

    local WASM_JS_NAME="genesis.js"
    local WASM_MODULE_NAME="genesis_bg.wasm"

    local ASSET_JS_PATH=${WASM_OUT_DIR_PATH}/${WASM_VERSION}/${WASM_JS_NAME}
    local ASSET_WASM_PATH=${WASM_OUT_DIR_PATH}/${WASM_VERSION}/${WASM_MODULE_NAME}

    gen_wasm_asset_metadata wasm${WASM_FEATURE}Script $WASM_VERSION $WASM_JS_NAME $ASSET_JS_PATH
    gen_wasm_asset_metadata wasm${WASM_FEATURE}Module $WASM_VERSION $WASM_MODULE_NAME $ASSET_WASM_PATH
}

gen_assets_metadata() {
    local ASSET_MANIFEST=$1

    local WASM_OUT_DIR_PATH=$2
    local WASM_SEQUENTIAL_VERSION=$3
    local WASM_PARALLEL_VERSION=$4

    mkdir -p $OUT_DIR
    echo "{\"assets\": []}" > $ASSETS_METADATA_PATH

    gen_data_assets_metadata $ASSET_MANIFEST
    gen_wasm_assets_metadata "Sequential" $WASM_SEQUENTIAL_VERSION $WASM_OUT_DIR_PATH
    gen_wasm_assets_metadata "Parallel" $WASM_PARALLEL_VERSION $WASM_OUT_DIR_PATH
}

gen_assets_metadata $1 $2 $3 $4
generate_dart_assets_manifest
