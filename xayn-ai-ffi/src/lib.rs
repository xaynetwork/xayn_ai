//! Common FFI components for the Xayn AI.
#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]

mod data;
mod error;

pub use crate::{
    data::{CDayOfWeek, CFeedback, CRelevance, CRerankMode, CUserAction},
    error::{CCode, Error},
};
