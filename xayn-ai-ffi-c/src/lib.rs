//! C FFI for the Xayn AI.
#![cfg_attr(
    doc,
    forbid(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)
)]
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod data;
pub mod reranker;
pub mod result;
mod slice;
pub mod utils;
