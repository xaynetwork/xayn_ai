//! C FFI for the Xayn AI.
#![forbid(
    unsafe_op_in_unsafe_fn,
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links
)]

pub mod data;
pub mod reranker;
pub mod result;
mod slice;
pub mod utils;

#[cfg(test)]
pub(crate) mod tests {
    /// Path to the current vocabulary file.
    pub const VOCAB: &str = "../data/rubert_v0001/vocab.txt";

    /// Path to the current onnx model file.
    pub const SMBERT_MODEL: &str = "../data/rubert_v0001/smbert.onnx";
}
