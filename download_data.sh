#!/bin/sh

set -eu

# We can't use `pushd` or `readlink -f` so we
# fall back to this.
CALLING_BASE_DIR="$(pwd -L)"

# path to the directory where this file is
SELF_DIR_PATH="$(dirname "$0")"

# in this way we can call the script from different directory
# but the data should go in the correct destination
DATA_DIR="$SELF_DIR_PATH/data"

CHECKSUM_FILE="sha256sums"

download()
{
  cd "$CALLING_BASE_DIR"
  TREAT_AS="$1"
  NAME="$2"
  VERSION="$3"
  ARCHIVE_BASENAME="${NAME}_$VERSION"
  ARCHIVE_NAME="$ARCHIVE_BASENAME.tgz"
  URL="http://s3-de-central.profitbricks.com/xayn-yellow-bert/$NAME/$ARCHIVE_NAME"

  curl "$URL" -o "$DATA_DIR/$ARCHIVE_NAME"

  cd "$DATA_DIR"
  tar -zxf "$ARCHIVE_NAME"

  # check content
  cd "$ARCHIVE_BASENAME"
  shasum -c "$CHECKSUM_FILE"

  if [ "$TREAT_AS" = "asset" ]; then
    # update symlinks (the `|| :` makes the shell ignore failure of rm even if `set -eu` is used)
    cd "$CALLING_BASE_DIR"
    rm "${SELF_DIR_PATH}/bindings/dart/example/assets/${NAME}"_v* || :
    ln -s "../../../../data/${ARCHIVE_BASENAME}" "${SELF_DIR_PATH}/bindings/dart/example/assets/${ARCHIVE_BASENAME}"
  fi
}

download asset smbert v0000
download asset qambert v0001
download asset ltr v0000
download misc ltr_feature_extraction_tests v0000
