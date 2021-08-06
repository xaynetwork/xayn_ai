use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ARCHIVE: &str = "";

/// Resolves the path to the MBert validation transcripts.
pub fn transcripts() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "ted_talk_transcripts.csv")
}
