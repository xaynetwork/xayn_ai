use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ARCHIVE: &str = "qambert_v0001";

/// Resolves the path to the QAMBert model.
pub fn model() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "qambert.onnx")
}

/// Resolves the path to the quantized QAMBert model.
pub fn model_quant() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "qambert-quant.onnx")
}

/// Resolves the path to the QAMBert vocabulary.
pub fn vocab() -> Result<PathBuf> {
    resolve_path(ARCHIVE, "vocab.txt")
}
