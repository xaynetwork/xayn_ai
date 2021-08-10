use std::{io::Result, path::PathBuf};

use crate::resolve_asset;

/// Resolves the path to the QAMBert vocabulary.
pub fn vocab() -> Result<PathBuf> {
    resolve_asset("qambertVocab")
}

/// Resolves the path to the QAMBert model.
pub fn model() -> Result<PathBuf> {
    resolve_asset("qambertModel")
}

/// Resolves the path to the quantized QAMBert model.
pub fn model_quant() -> Result<PathBuf> {
    Ok(resolve_asset("qambertModel")?.with_file_name("qambert-quant.onnx"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab() {
        assert!(vocab().is_ok());
    }

    #[test]
    fn test_model() {
        assert!(model().is_ok());
    }
}
