//! Mobile FFI for Xayn AI.
#![allow(unused_unsafe)]

mod ai;
mod systems;
mod utils;

#[cfg(doc)]
pub use self::ai::{xaynai_drop, xaynai_new, xaynai_rerank};
