use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ARCHIVE: &str = "ltr_feature_extraction_tests_v0000";

/// Resolves the path to the LTR feature extraction test data.
pub fn data() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "")
}
