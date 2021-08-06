use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ARCHIVE: &str = "ltr_v0000";

/// Resolves the path to the LTR model.
pub fn model() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "ltr.binparams")
}
