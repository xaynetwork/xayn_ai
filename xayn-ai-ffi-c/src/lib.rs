//! C FFI for the Xayn AI.
#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]
#![allow(unused_unsafe)]

pub mod data;
pub mod reranker;
pub mod result;
mod slice;
pub mod utils;

#[cfg(test)]
pub(crate) mod tests {
    /// Path to the current vocabulary file.
    pub const VOCAB: &str = "../data/smbert_v0000/vocab.txt";

    /// Path to the current smbert onnx model file.
    pub const SMBERT_MODEL: &str = "../data/smbert_v0000/smbert.onnx";

    /// Path to the current qambert onnx model file.
    pub const QAMBERT_MODEL: &str = "../data/qambert_v0000/qambert.onnx";
}
