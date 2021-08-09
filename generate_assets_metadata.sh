#!/bin/bash

# Generates the metadata of the assets (ai and wasm). If an asset of the `ai_assets` array
# contains a `chunk_size` key, the script splits the asset into chunks where each chunk has
# a maximum size of `chunk_size`. The format of the `chunk_size` value is equivalent to the
# `SIZE` argument in `split` or `gsplit` on macOS. See `split`/`gsplit` man page for more details.
#
# The script needs to be executed in the root of the repository.
#
# Usage:
# ./generate_assets_metadata <path of assets_manifest.json> [<wasm version>] [<path of the wasm output directory>]

set -e

OUT_DIR="out"
ASSETS_METADATA_PATH=$OUT_DIR/assets_metadata.json

if [ -z ${GITHUB_ACTIONS} ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command gsplit --version /dev/null; then
            SPLIT="gsplit"
        else
            echo "Requires the GNU version of 'split'. Use 'brew install coreutils' to install it."
            exit 1
        fi
    else
        SPLIT="split"
    fi
else
    if [ $RUNNER_OS == "macOS" ]; then
        if command gsplit --version /dev/null; then
            SPLIT="gsplit"
        else
            echo "Cannot find 'gsplit'."
            exit 1
        fi
    else
        SPLIT="split"
    fi
fi

if ! command gomplate -v /dev/null; then
    echo "Cannot find 'gomplate'."
    exit 1
fi

if ! command jq --version /dev/null; then
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

gen_ai_assets_metadata() {
    local ASSET_MANIFEST=$1
    local TMP_FILE=$(mktemp)
    local CHUNKS_DIR=$OUT_DIR/chunks

    $(rm -rf $CHUNKS_DIR || true) && mkdir $CHUNKS_DIR

    for ASSET in $(cat $ASSET_MANIFEST | jq -c '.ai_assets[]'); do
        local ASSET_PATH=$(echo $ASSET | jq -r '.path')
        local ASSET_CHECKSUM=$(calc_checksum $ASSET_PATH)
        local ASSET_WITH_CHECKSUM=$(echo $ASSET | jq -c --arg checksum $ASSET_CHECKSUM '. |= .+ {"checksum": $checksum}')

        local UPDATED_ASSET=$ASSET_WITH_CHECKSUM
        local ASSET_CHUNK_SIZE=$(echo $ASSET | jq -r '.chunk_size')
        if [ "$ASSET_CHUNK_SIZE" != "null" ]; then
            local ASSET_URL_SUFFIX=$(echo $ASSET | jq -r '.url_suffix')
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

gen_wasm_assets_metadata() {
    local ASSET_MANIFEST=$1
    local WASM_VERSION=$2
    local WASM_OUT_DIR_PATH=$3
    local TMP_FILE=$(mktemp)

    for WASM_ASSET in $(cat $ASSET_MANIFEST | jq -c '.wasm_assets[]'); do
        local ASSET_FILENAME=$(echo $WASM_ASSET | jq -r '.filename')
        local ASSET_PATH=${WASM_OUT_DIR_PATH}/${ASSET_FILENAME}

        if [ -f "$ASSET_PATH" ]; then
            local ASSET_CHECKSUM=$(calc_checksum $ASSET_PATH)
            local ASSET_WITH_CHECKSUM=$(echo $WASM_ASSET | jq -c --arg checksum $ASSET_CHECKSUM '. |= .+ {"checksum": $checksum}')
            local ASSET_URL_SUFFIX=${WASM_VERSION}/${ASSET_FILENAME}
            ASSET=$(echo $ASSET_WITH_CHECKSUM | jq -c --arg path $ASSET_PATH --arg url_suffix $ASSET_URL_SUFFIX '. |= .+ {"path": $path, "url_suffix": $url_suffix}')
        else
            ASSET=$(echo $WASM_ASSET | jq -c '. |= .+ {"url_suffix": "", "checksum": ""}')
        fi

        local UPDATED_ASSET=$(echo $ASSET | jq -c '. |= .+ {"fragments": []}' | jq -c 'del(.filename)')

        jq --argjson wasm_asset $UPDATED_ASSET '.assets |= .+ [$wasm_asset]' $ASSETS_METADATA_PATH > $TMP_FILE
        mv $TMP_FILE $ASSETS_METADATA_PATH
    done
}

gen_assets_metadata() {
    local ASSET_MANIFEST=$1
    local WASM_VERSION=$2
    local WASM_OUT_DIR_PATH=$3

    mkdir -p $OUT_DIR
    echo "{\"assets\": []}" > $ASSETS_METADATA_PATH

    gen_ai_assets_metadata $ASSET_MANIFEST
    gen_wasm_assets_metadata $ASSET_MANIFEST $WASM_VERSION $WASM_OUT_DIR_PATH
}

gen_assets_metadata $1 $2 $3
generate_dart_assets_manifest
