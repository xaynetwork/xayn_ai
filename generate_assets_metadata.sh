#!/bin/bash

# Generates the metadata of the assets (ai and wasm). If an asset of the `data_assets` array
# contains a `chunk_size` key, the script splits the asset into chunks where each chunk has
# a maximum size of `chunk_size`. The format of the `chunk_size` value is equivalent to the
# `SIZE` argument in `split` or `gsplit` on macOS. See `split`/`gsplit` man page for more details.
#
# The script needs to be executed in the root of the repository.
#
# Usage:
# ./generate_assets_metadata <path of assets_manifest.json> [<path of the wasm output directory>]
set -e

source $(dirname "$0")/scripts/assets_generation_lib.sh

gen_data_and_wasm_assets_metadata "$1" "$2"
generate_dart_assets_manifest
