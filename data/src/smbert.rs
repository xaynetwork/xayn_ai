use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ARCHIVE: &str = "smbert_v0000";

/// Resolves the path to the SMBert model.
pub fn model() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "smbert.onnx")
}

/// Resolves the path to the quantized SMBert model.
pub fn model_quant() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "smbert-quant.onnx")
}

/// Resolves the path to the SMBert vocabulary.
pub fn vocab() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "vocab.txt")
}
