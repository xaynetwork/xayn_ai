use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ASSET: &str = "ltr_test_data_v0000";

/// Resolves the path to the LTR feature extraction test data.
pub fn data() -> Result<PathBuf> {
    resolve_path(&["data", ASSET, "feature_extraction"])
}

/// Resolves the path to the intermediate LTR test model.
pub fn intermediate() -> Result<PathBuf> {
    resolve_path(&["data", ASSET, "check_training_intermediates.binparams"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data() {
        assert!(data().is_ok());
    }

    #[test]
    fn test_intermediate() {
        assert!(intermediate().is_ok());
    }
}
