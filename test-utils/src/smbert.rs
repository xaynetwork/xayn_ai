use std::{io::Result, path::PathBuf};

use crate::asset::resolve_asset;

/// Resolves the path to the SMBert vocabulary.
pub fn vocab() -> Result<PathBuf> {
    resolve_asset("smbertVocab")
}

/// Resolves the path to the SMBert model.
pub fn model() -> Result<PathBuf> {
    resolve_asset("smbertModel")
}

/// Resolves the path to the quantized SMBert model.
pub fn model_quant() -> Result<PathBuf> {
    Ok(resolve_asset("smbertModel")?.with_file_name("smbert-quant.onnx"))
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
