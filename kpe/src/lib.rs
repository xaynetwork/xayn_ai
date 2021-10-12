#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]
#![allow(dead_code)]

mod tokenizer;

#[cfg(doc)]
pub use crate::tokenizer::{key_phrase::RankedKeyPhrases, TokenizerError};
