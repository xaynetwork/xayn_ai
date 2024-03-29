#!/bin/bash

# Generates the metadata of the assets (ai and wasm). If an asset of the `data_assets` array
# contains a `chunk_size` key, the script splits the asset into chunks where each chunk has
# a maximum size of `chunk_size`. The format of the `chunk_size` value is equivalent to the
# `SIZE` argument in `split` or `gsplit` on macOS. See `split`/`gsplit` man page for more details.

OUT_DIR="$(dirname "$BASH_SOURCE")/../out"
ASSETS_METADATA_PATH="$OUT_DIR/assets_metadata.json"
WASM_SCRIPT_NAME="genesis.js"
WASM_MODULE_NAME="genesis_bg.wasm"
WEB_WORKER_NAME="worker.js"

set -e

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

    mkdir "${CHUNKS_DIR}/${ASSET_VERSION}"

    $SPLIT --numeric-suffixes=0 -b $ASSET_CHUNK_SIZE "$ASSET_PATH" "${CHUNKS_DIR}/${ASSET_VERSION}/${ASSET_FILENAME}_${ASSET_CHUNK_SIZE}_"
}

generate_dart_assets_manifest() {
    gomplate -d assets_manifest="$ASSETS_METADATA_PATH" -f data/asset_templates/base_assets.dart.tmpl -o bindings/dart/lib/src/common/reranker/assets.dart
    flutter format bindings/dart/lib/src/common/reranker/assets.dart

    gomplate -d assets_manifest="$ASSETS_METADATA_PATH" -f data/asset_templates/web_assets.dart.tmpl -o bindings/dart/lib/src/web/reranker/assets.dart
    flutter format bindings/dart/lib/src/web/reranker/assets.dart
}

calc_checksum() {
    echo $(shasum -a 256 $1 | awk '{ print $1 }')
}

gen_data_assets_metadata() {
    local ASSET_MANIFEST=$1
    local DATA_DIR="$(dirname "$BASH_SOURCE")/../data"
    local CHUNKS_DIR="$DATA_DIR/chunks"
    local TMP_FILE=$(mktemp)

    $(rm -rf $CHUNKS_DIR || true) && mkdir $CHUNKS_DIR

    for ASSET in $(cat "$ASSET_MANIFEST" | jq -c '.data_assets[]'); do

        local ASSET_URL_SUFFIX=$(echo $ASSET | jq -r '.url_suffix')
        local ASSET_PATH="$DATA_DIR/$ASSET_URL_SUFFIX"
        local ASSET_CHECKSUM=$(calc_checksum $ASSET_PATH)
        local ASSET_WITH_CHECKSUM=$(echo $ASSET | jq -c --arg checksum $ASSET_CHECKSUM '. |= .+ {"checksum": $checksum}')

        local UPDATED_ASSET=$ASSET_WITH_CHECKSUM
        local ASSET_CHUNK_SIZE=$(echo $ASSET | jq -r '.chunk_size')
        if [ "$ASSET_CHUNK_SIZE" != "null" ]; then
            local ASSET_FILENAME=$(basename $ASSET_URL_SUFFIX)
            local ASSET_VERSION=$(dirname $ASSET_URL_SUFFIX)

            split_asset_into_chunks "$CHUNKS_DIR" $ASSET_CHUNK_SIZE "$ASSET_PATH" "$ASSET_FILENAME" "$ASSET_VERSION"

            local FRAGMENTS=""
            for CHUNK_PATH in $(find "${CHUNKS_DIR}/${ASSET_VERSION}" -name "${ASSET_FILENAME}_*" | sort -n); do
                local FRAGMENT_CHECKSUM=$(calc_checksum "$CHUNK_PATH")
                local FRAGMENT_FILENAME=$(basename "$CHUNK_PATH")
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

        jq --argjson asset $UPDATED_ASSET '.assets |= .+ [$asset]' "$ASSETS_METADATA_PATH" > "$TMP_FILE"
        mv "$TMP_FILE" "$ASSETS_METADATA_PATH"
    done
}

gen_web_asset_metadata() {
    local ASSET_PATH=$1
    local WASM_VERSION=$2

    local ASSET="{}"
    local ASSET_CHECKSUM=$(calc_checksum "$ASSET_PATH")
    local ASSET_WITH_CHECKSUM=$(echo $ASSET | jq -c --arg checksum $ASSET_CHECKSUM '. |= .+ {"checksum": $checksum}')
    local ASSET_FILENAME=$(basename "$ASSET_PATH")
    local ASSET_URL_SUFFIX="$WASM_VERSION/$ASSET_FILENAME"
    ASSET=$(echo $ASSET_WITH_CHECKSUM | jq -c --arg url_suffix $ASSET_URL_SUFFIX '. |= .+ {"url_suffix": $url_suffix}')

    echo $ASSET
}

gen_web_worker_asset_metadata() {
    local WEB_WORKER_OUT_DIR_PATH=$1
    local WASM_VERSION=$2

    local ASSET_WEB_WORKER_PATH="$WEB_WORKER_OUT_DIR_PATH/$WEB_WORKER_NAME"
    local ASSET_WEB_WORKER=$(gen_web_asset_metadata "$ASSET_WEB_WORKER_PATH" "$WASM_VERSION")

    add_to_upload_list "$ASSET_WEB_WORKER_PATH" "$ASSET_WEB_WORKER"

    echo $ASSET_WEB_WORKER
}

# Generates and adds the following object to the `wasm_assets` object.
# Furthermore, the script, the module and any additional javascript file
# will be added to the `upload` list.
#
# "<feature>": {
#   "script": {
#     "checksum": "<checksum>",
#     "url_suffix": "<version>/<filename>"
#   },
#   "module": {
#     "checksum": "<checksum>",
#     "url_suffix": "<version>/<filename>"
#   },
#   "web_worker": {
#     "checksum": "<checksum>",
#     "url_suffix": "<version>/<filename>"
#   }
# },
gen_wasm_assets_metadata() {
    local WASM_OUT_DIR_PATH=$1
    local WASM_VERSION=$2
    local WASM_FEATURE=$3
    local ASSET_WEB_WORKER=$4

    local ASSET_JS_PATH="$WASM_OUT_DIR_PATH/$WASM_VERSION/$WASM_SCRIPT_NAME"
    local ASSET_WASM_PATH="$WASM_OUT_DIR_PATH/$WASM_VERSION/$WASM_MODULE_NAME"

    local WASM_PACKAGE="{}"
    local ASSET_JS=$(gen_web_asset_metadata "$ASSET_JS_PATH" "$WASM_VERSION")
    WASM_PACKAGE=$(echo $WASM_PACKAGE | jq -c --argjson wasm_script $ASSET_JS '. |= .+ {"script": $wasm_script}')
    local ASSET_WASM=$(gen_web_asset_metadata "$ASSET_WASM_PATH" "$WASM_VERSION")
    WASM_PACKAGE=$(echo $WASM_PACKAGE | jq -c --argjson wasm_module $ASSET_WASM '. |= .+ {"module": $wasm_module}')
    WASM_PACKAGE=$(echo $WASM_PACKAGE | jq -c --argjson web_worker_script $ASSET_WEB_WORKER '. |= .+ {"web_worker": $web_worker_script}')

    local TMP_FILE=$(mktemp)
    jq --argjson wasm_asset $WASM_PACKAGE --arg feature "$WASM_FEATURE" '.wasm_assets |= .+ {($feature): $wasm_asset}' "$ASSETS_METADATA_PATH" > "$TMP_FILE"
    mv "$TMP_FILE" "$ASSETS_METADATA_PATH"

    add_to_upload_list "$ASSET_JS_PATH" "$ASSET_JS"
    add_to_upload_list "$ASSET_WASM_PATH" "$ASSET_WASM"

    for ASSET_PATH in $(find "${WASM_OUT_DIR_PATH}/${WASM_VERSION}" -type f -name '*.js' ! -name $WASM_SCRIPT_NAME ! -name $WEB_WORKER_NAME); do
        local ASSET_FILENAME=$(basename "$ASSET_PATH")
        local ASSET_URL_SUFFIX="${WASM_VERSION}/${ASSET_FILENAME}"
        add_to_upload_list "$ASSET_PATH" "{\"url_suffix\": \"$ASSET_URL_SUFFIX\"}"
    done
}

# Generates for the given asset the following object and adds it to the `upload` list.
#
# {
#   "url_suffix": "<version>/<filename>",
#   "path": "<path>"
# },
#
# The asset can only be a file. If the asset does not exist, the script will
# exit with an error.
add_to_upload_list() {
    local ASSET_PATH=$1
    local ASSET_META=$2
    if [ -f "$ASSET_PATH" ]; then
        local TMP_FILE=$(mktemp)
        local ASSET_URL_SUFFIX=$(echo $ASSET_META | jq -c '. | {"url_suffix": .url_suffix}')

        local ASSET_URL_SUFFIX_PATH=$(echo $ASSET_URL_SUFFIX | jq -c --arg path "$ASSET_PATH" '. |= .+ {"path": $path}')
        jq --argjson upload $ASSET_URL_SUFFIX_PATH '.upload |= .+ [$upload]' "$ASSETS_METADATA_PATH" > "$TMP_FILE"
        mv "$TMP_FILE" "$ASSETS_METADATA_PATH"
    else
        echo "$ASSET_PATH does not exist"
        exit 1
    fi
}

gen_data_and_wasm_assets_metadata() {
    local ASSET_MANIFEST=$1
    local WASM_OUT_DIR_PATH=$2

    mkdir -p "$OUT_DIR"
    echo "{\"assets\": [], \"wasm_assets\": {}}" > "$ASSETS_METADATA_PATH"

    gen_data_assets_metadata "$ASSET_MANIFEST"

    if [ -d "$WASM_OUT_DIR_PATH" ]; then
        for WASM_VERSION in $(find "$WASM_OUT_DIR_PATH" -type f -maxdepth 2 -name 'package.json' -exec sh -c "dirname {} | xargs basename" \;); do
            local PACKAGE="$WASM_OUT_DIR_PATH/$WASM_VERSION/package.json"
            local WASM_FEATURE=$(cat "$PACKAGE" | jq -r '.feature')
            local ASSET_WEB_WORKER=$(gen_web_worker_asset_metadata "$WASM_OUT_DIR_PATH/$WASM_VERSION" "$WASM_VERSION")
            gen_wasm_assets_metadata "$WASM_OUT_DIR_PATH" "$WASM_VERSION" "$WASM_FEATURE" "$ASSET_WEB_WORKER"
        done
    fi
}

gen_data_assets_metadata_only() {
    local ASSET_MANIFEST=$1

    mkdir -p "$OUT_DIR"
    echo "{\"assets\": []}" > "$ASSETS_METADATA_PATH"

    gen_data_assets_metadata "$ASSET_MANIFEST"
    echo "$ASSETS_METADATA_PATH"
}

