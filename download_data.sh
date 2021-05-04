#!/bin/sh

# path to this file
SELF_PATH=$(readlink -f "$0")
# path to the directory where this file is
SELF_DIR_PATH=$(dirname "$SELF_PATH")

# in this way we can call the script from different directory
# but the data should go in the correct destination
DATA_DIR="$SELF_DIR_PATH/data/"

download_rubert()
{
  VERSION=$1
  ARCHIVE="rubert_${VERSION}.tgz"
  URL="http://s3-de-central.profitbricks.com/xayn-yellow-bert/rubert/$ARCHIVE"

  curl $URL -o $DATA_DIR/$ARCHIVE

  cd $DATA_DIR
  tar -zxf $ARCHIVE
}

download_rubert v0001

