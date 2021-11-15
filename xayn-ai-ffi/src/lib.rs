//! Common FFI components for the Xayn AI.
#![cfg_attr(
    doc,
    forbid(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)
)]
#![forbid(unsafe_op_in_unsafe_fn)]

mod error;

pub use crate::error::{CCode, Error};
