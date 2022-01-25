//! WASM FFI for the Xayn AI.
#![cfg_attr(
    doc,
    forbid(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)
)]
#![forbid(unsafe_op_in_unsafe_fn)]
// TODO: remove clippy lint once wasm-bindgen fixes the regression from 0.2.79
#![allow(clippy::unused_unit)]

#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
use ::{
    js_sys::Promise,
    wasm_bindgen::{prelude::wasm_bindgen, JsValue},
};

#[cfg(not(tarpaulin))]
mod ai;
#[cfg(not(tarpaulin))]
mod error;

#[cfg(all(not(tarpaulin), doc))]
pub use crate::ai::WXaynAi;

/// Reexport to allow initialization of a WebWorker based on the rayon thread pool.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Stub which is used when the wasm blob was compiled without the `multithreaded` feature.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
#[wasm_bindgen(js_name = initThreadPool)]
pub fn init_thread_pool(_num_threads: usize) -> Promise {
    Promise::resolve(&JsValue::UNDEFINED)
}
