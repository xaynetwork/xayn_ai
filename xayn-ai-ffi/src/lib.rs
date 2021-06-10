//! Common FFI components for the Xayn AI.
#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]

mod data;
mod error;

pub use data::CRerankMode;
pub use crate::error::{CCode, Error};

pub use xayn_ai::{DayOfWeek, Relevance, UserAction, UserFeedback};
