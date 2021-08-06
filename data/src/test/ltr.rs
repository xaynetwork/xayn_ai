use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ARCHIVE: &str = "ltr_test_data_v0000";

/// Resolves the path to the LTR feature extraction test data.
pub fn data() -> Result<PathBuf> {
    resolve_path(PathBuf::from(ARCHIVE).join("feature_extraction"), "")
}

/// Resolves the path to the intermediate LTR test model.
pub fn intermediate() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "check_training_intermediates.binparams")
}
