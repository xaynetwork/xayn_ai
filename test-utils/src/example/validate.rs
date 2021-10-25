use std::{io::Result, path::PathBuf};

use crate::asset::{resolve_path, DATA_DIR};

const ASSET: &str = "ted_talk_transcripts.csv";

/// Resolves the path to the MBert validation transcripts.
pub fn transcripts() -> Result<PathBuf> {
    resolve_path(&[DATA_DIR, ASSET])
}
