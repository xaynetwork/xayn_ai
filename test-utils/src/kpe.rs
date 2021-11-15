use std::{io::Result, path::PathBuf};

use crate::asset::{resolve_path, DATA_DIR};

const ASSET: &str = "kpe_v0000";

/// Resolves the path to the Bert vocabulary.
pub fn vocab() -> Result<PathBuf> {
    resolve_path(&[DATA_DIR, ASSET, "vocab.txt"])
}

/// Resolves the path to the Bert model.
pub fn bert() -> Result<PathBuf> {
    resolve_path(&[DATA_DIR, ASSET, "bert-quantized.onnx"])
}

/// Resolves the path to the CNN model.
pub fn cnn() -> Result<PathBuf> {
    resolve_path(&[DATA_DIR, ASSET, "cnn.binparams"])
}

/// Resolves the path to the Classifier model.
pub fn classifier() -> Result<PathBuf> {
    resolve_path(&[DATA_DIR, ASSET, "classifier.binparams"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab() {
        assert!(vocab().is_ok());
    }

    #[test]
    fn test_bert() {
        assert!(bert().is_ok());
    }

    #[test]
    fn test_cnn() {
        assert!(cnn().is_ok());
    }

    #[test]
    fn test_classifier() {
        assert!(classifier().is_ok());
    }
}
