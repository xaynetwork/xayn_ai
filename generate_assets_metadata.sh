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
        local ASSET_WITH_CHECKSUM=$(echo $ASSET | jq -c --arg checksum $ASSET_CHECKSUM '. |= .+ {"checksum": $checksum}')

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
                local FRAGMENT="{\"url_suffix\": \"${ASSET_VERSION}/${FRAGMENT_FILENAME}\", \"checksum\": \"$FRAGMENT_CHECKSUM\"}"
                if [ -z "${FRAGMENTS}" ]; then
                    FRAGMENTS="$FRAGMENT"
                else
                    FRAGMENTS+=", $FRAGMENT"
                fi
                add_to_upload_list "$CHUNK_PATH" "$FRAGMENT"
            done

            local UPDATED_ASSET=$(echo $ASSET_WITH_CHECKSUM | jq -c --argjson fragments "[$FRAGMENTS]" '. |= .+ {"fragments": $fragments}' | jq -c 'del(.chunk_size)')
        else
            local UPDATED_ASSET=$(echo $ASSET_WITH_CHECKSUM | jq -c '. |= .+ {"fragments": []}')
            add_to_upload_list "$ASSET_PATH" "$UPDATED_ASSET"
        fi

        jq --argjson asset $UPDATED_ASSET '.assets |= .+ [$asset]' $ASSETS_METADATA_PATH > $TMP_FILE
        mv $TMP_FILE $ASSETS_METADATA_PATH
    done
}

gen_wasm_asset_metadata() {
    local WASM_VERSION=$1
    local ASSET_PATH=$2

    local ASSET="{}"

    if [ -f "$ASSET_PATH" ]; then
        local ASSET_CHECKSUM=$(calc_checksum $ASSET_PATH)
        local ASSET_WITH_CHECKSUM=$(echo $ASSET | jq -c --arg checksum $ASSET_CHECKSUM '. |= .+ {"checksum": $checksum}')
        local ASSET_FILENAME=$(basename $ASSET_PATH)
        local ASSET_URL_SUFFIX=${WASM_VERSION}/${ASSET_FILENAME}
        ASSET=$(echo $ASSET_WITH_CHECKSUM | jq -c --arg url_suffix $ASSET_URL_SUFFIX '. |= .+ {"url_suffix": $url_suffix}')
    else
        ASSET=$(echo $ASSET | jq -c '. |= .+ {"url_suffix": "", "checksum": ""}')
    fi

    echo $ASSET
}

# Generates and adds the following object to the `wasm_assets` array.
# Furthermore, any asset (script, module or snippets) will be added to
# the `upload` list if it exists.
#
# {
#   "feature": "<feature>",
#   "script": {
#     "checksum": "<checksum>",
#     "url_suffix": "<version>/<filename>"
#   },
#   "module": {
#     "checksum": "<checksum>",
#     "url_suffix": "<version>/<filename>"
#   }
# },
gen_wasm_assets_metadata() {
    local WASM_FEATURE=$1
    local WASM_VERSION=$2
    local WASM_OUT_DIR_PATH=$3

    local ASSET_JS_PATH=${WASM_OUT_DIR_PATH}/${WASM_VERSION}/genesis.js
    local ASSET_WASM_PATH=${WASM_OUT_DIR_PATH}/${WASM_VERSION}/genesis_bg.wasm

    local WASM_PACKAGE="{\"feature\": \"$WASM_FEATURE\"}"
    local ASSET_JS=$(gen_wasm_asset_metadata $WASM_VERSION $ASSET_JS_PATH)
    WASM_PACKAGE=$(echo $WASM_PACKAGE | jq -c --argjson script $ASSET_JS '. |= .+ {"script": $script}')
    local ASSET_WASM=$(gen_wasm_asset_metadata $WASM_VERSION $ASSET_WASM_PATH)
    WASM_PACKAGE=$(echo $WASM_PACKAGE | jq -c --argjson wasm_module $ASSET_WASM '. |= .+ {"module": $wasm_module}')

    local TMP_FILE=$(mktemp)
    jq --argjson wasm_asset $WASM_PACKAGE '.wasm_assets |= .+ [$wasm_asset]' $ASSETS_METADATA_PATH > $TMP_FILE
    mv $TMP_FILE $ASSETS_METADATA_PATH

    add_to_upload_list "$ASSET_JS_PATH" "$ASSET_JS"
    add_to_upload_list "$ASSET_WASM_PATH" "$ASSET_WASM"

    # The S3 URI must end with '/' in order to upload a directory.
    # We don't have to add the `snippets` folder name to the snippets url suffix
    # because s3cmd will already take the name from the snippets path.
    local SNIPPETS_URL_SUFFIX=$WASM_VERSION/
    local SNIPPETS_PATH=$WASM_OUT_DIR_PATH/$WASM_VERSION/snippets

    add_to_upload_list "$SNIPPETS_PATH" "{\"url_suffix\": \"$SNIPPETS_URL_SUFFIX\"}"
}

# If the given asset exists (file or directory), the function will generate the
# following object for the asset and add it to the `upload` list.
#
# {
#   "url_suffix": "<version>/<filename/dirname>",
#   "path": "<path>"
# },
add_to_upload_list() {
    local ASSET_PATH=$1
    local ASSET_META=$2

    if [ -f "$ASSET_PATH" ] || [ -d "$ASSET_PATH" ]; then
        local TMP_FILE=$(mktemp)
        local ASSET_URL_SUFFIX=$(echo $ASSET_META | jq -c '. | {"url_suffix": .url_suffix}')

        local ASSET_URL_SUFFIX_PATH=$(echo $ASSET_URL_SUFFIX | jq -c --arg path $ASSET_PATH '. |= .+ {"path": $path}')
        jq --argjson upload $ASSET_URL_SUFFIX_PATH '.upload |= .+ [$upload]' $ASSETS_METADATA_PATH > $TMP_FILE
        mv $TMP_FILE $ASSETS_METADATA_PATH
    fi
}

gen_assets_metadata() {
    local ASSET_MANIFEST=$1

    local WASM_OUT_DIR_PATH=$2
    local WASM_SEQUENTIAL_VERSION=$3
    local WASM_PARALLEL_VERSION=$4

    mkdir -p $OUT_DIR
    echo "{\"assets\": []}" > $ASSETS_METADATA_PATH

    gen_data_assets_metadata $ASSET_MANIFEST
    gen_wasm_assets_metadata "" $WASM_SEQUENTIAL_VERSION $WASM_OUT_DIR_PATH
    gen_wasm_assets_metadata "Parallel" $WASM_PARALLEL_VERSION $WASM_OUT_DIR_PATH
}

gen_assets_metadata "$1" "$2" "$3" "$4"
generate_dart_assets_manifest
