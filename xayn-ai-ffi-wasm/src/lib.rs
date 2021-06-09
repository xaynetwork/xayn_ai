//! WASM FFI for the Xayn AI.
#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]

#[cfg(not(tarpaulin))]
mod ai;
#[cfg(not(tarpaulin))]
mod error;
#[cfg(not(tarpaulin))]
mod history;

#[cfg(all(not(tarpaulin), doc))]
pub use crate::ai::WXaynAi;

