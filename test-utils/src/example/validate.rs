use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ASSET: &str = "ted_talk_transcripts.csv";

/// Resolves the path to the MBert validation transcripts.
pub fn transcripts() -> Result<PathBuf> {
    resolve_path(&["data", ASSET])
}
