#!/bin/sh
# This script takes as input a directory, the name of the archive and a version.
# It creates an archive in the correct format in the current directory
# and adds the necessary information to verify its content.
# The archive will contain the directory name and the provided version.
# If the option --upload is provided the script will upload the archive to the s3 bucket.

# ./prepare_data.sh ./bert_models rubert v0 will generate an archive rubert_v0.tgz
# with one directory rubert_v0 that contains the files that are present in ./bert_models.

# directory to prepare for upload
DIR_PATH=$1
shift
NAME=$1
shift
VERSION=$1
shift

while [ $# -gt 0 ]; do
  opt="$1"
  shift

  case $opt in
    --upload)
    UPLOAD=true
    ;;
  esac
done

DIR_PATH=$(pwd)/$DIR_PATH
DIR_NAME=$(basename $DIR_PATH)
ARCHIVE_BASENAME="${NAME}_$VERSION"
ARCHIVE_NAME="$ARCHIVE_BASENAME.tgz"
URL="s3://xayn-yellow-bert/$NAME/$ARCHIVE_NAME"
CHECKSUM_FILE="sha256sums"

CURRENT_DIR=$(pwd)
cd $DIR_PATH

# create a directory with the expected name
TMP_DIR=$(mktemp -d)
cd $TMP_DIR
cp -r $DIR_PATH .
if [ $DIR_NAME != $ARCHIVE_BASENAME ]; then
  mv $DIR_NAME $ARCHIVE_BASENAME
fi
TO_ARCHIVE="$TMP_DIR/$ARCHIVE_BASENAME"

# compute checksum file
cd $ARCHIVE_BASENAME
rm -f $CHECKSUM_FILE
find . -type f -not -iname $CHECKSUM_FILE -not -name ".DS_Store" -print0 | xargs -0 shasum -a 256 > $CHECKSUM_FILE

cd $CURRENT_DIR

# prepare archive
tar czf $ARCHIVE_NAME --exclude ".DS_Store" -C $TMP_DIR $ARCHIVE_BASENAME
rm -rf $TMP_DIR

if [ "$UPLOAD" = true ]; then
  s3cmd put --acl-public --guess-mime-type $ARCHIVE_NAME $URL
fi
