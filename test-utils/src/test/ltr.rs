use std::{io::Result, path::PathBuf};

use crate::asset::{resolve_path, DATA_DIR};

const ASSET: &str = "ltr_test_data_v0001";

/// Resolves the path to the LTR feature extraction test data.
pub fn feature_extraction_test_cases() -> Result<PathBuf> {
    resolve_path(&[DATA_DIR, ASSET, "feature_extraction"])
}

/// Resolves the path to the intermediate LTR test model.
pub fn training_intermediates() -> Result<PathBuf> {
    resolve_path(&[DATA_DIR, ASSET, "check_training_intermediates.binparams"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction_test_cases() {
        assert!(feature_extraction_test_cases().is_ok());
    }

    #[test]
    fn test_training_intermediates() {
        assert!(training_intermediates().is_ok());
    }
}
