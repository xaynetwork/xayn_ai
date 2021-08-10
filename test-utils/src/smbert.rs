use std::{
    io::{Error, ErrorKind, Result},
    path::PathBuf,
};

use crate::resolve_asset;

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
    Ok(resolve_asset("smbertModel")?
        .parent()
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "missing asset 'smbert'"))?
        .join("smbert-quant.onnx"))
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
