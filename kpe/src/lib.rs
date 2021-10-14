#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]
#![allow(dead_code)]

mod model;
mod tokenizer;

#[cfg(doc)]
pub use crate::{
    model::ModelError,
    tokenizer::{key_phrase::RankedKeyPhrases, TokenizerError},
};
