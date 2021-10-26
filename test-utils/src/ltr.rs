use std::{io::Result, path::PathBuf};

use crate::asset::resolve_asset;

/// Resolves the path to the LTR model.
pub fn model() -> Result<PathBuf> {
    resolve_asset("ltrModel")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model() {
        assert!(model().is_ok());
    }
}
