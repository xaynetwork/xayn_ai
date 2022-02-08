#!/bin/bash

# Generates the metadata of the data assets. If an asset of the `data_assets` array
# contains a `chunk_size` key, the script splits the asset into chunks where each chunk has
# a maximum size of `chunk_size`. The format of the `chunk_size` value is equivalent to the
# `SIZE` argument in `split` or `gsplit` on macOS. See `split`/`gsplit` man page for more details.
#
# Usage:
# ./upload_assets <path of assets_manifest.json> <bucket url e.g. s3://xayn-yellow-bert>
set -e

source $(dirname "$0")/assets_generation_lib.sh

ASSET_METADATA=$(gen_data_assets_metadata_only "$1")
BUCKET_URL="$2"

for ASSET in $(cat "$ASSET_METADATA" | jq -c '.upload[]'); do
    ASSET_URL_SUFFIX=$(echo $ASSET | jq -r '.url_suffix')
    ASSET_PATH=$(echo $ASSET | jq -r '.path')
    s3cmd sync -v --acl-public --guess-mime-type --no-mime-magic --skip-existing $ASSET_PATH ${BUCKET_URL}/$ASSET_URL_SUFFIX
done

ASSET_MANIFEST="$(dirname "$ASSET_METADATA")/asset_manifest.json"
jq 'del(.upload)' "$ASSET_METADATA" > $ASSET_MANIFEST

echo "assets manifest path: $ASSET_MANIFEST"
