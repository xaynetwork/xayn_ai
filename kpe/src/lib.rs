//! The KPE pipeline extracts key phrases from a sequence.
//!
//! See `examples/` for a usage example.
#![cfg_attr(
    doc,
    forbid(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)
)]
#![forbid(unsafe_op_in_unsafe_fn)]

mod config;
mod model;
mod pipeline;
mod tokenizer;

pub use crate::{
    config::{Configuration, ConfigurationError},
    pipeline::{Pipeline, PipelineError},
    tokenizer::key_phrase::RankedKeyPhrases,
};

#[cfg(doc)]
pub use crate::{model::ModelError, tokenizer::TokenizerError};
