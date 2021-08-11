//! WASM FFI for the Xayn AI.
#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]

#[cfg(not(tarpaulin))]
mod ai;
#[cfg(not(tarpaulin))]
mod error;

pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(all(not(tarpaulin), doc))]
pub use crate::ai::WXaynAi;
