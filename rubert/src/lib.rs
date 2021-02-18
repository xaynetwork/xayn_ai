#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]
//! The RuBert pipeline computes embeddings of sentences.
//!
//! Sentences are anything string-like and can also be single words or snippets. The embeddings are
//! f32-arrays and their shape depends on the pooling strategy.
//!
//! ```no_run
//! use rubert::{Builder, Pooler};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let rubert = Builder::new("vocab.txt", "model.onnx")
//!         .with_strip_accents(true)
//!         .with_lowercase(true)
//!         .with_batch_size(10)?
//!         .with_token_size(64)?
//!         .with_pooling(Pooler::First)
//!         .build()?;
//!
//!     let sentences = vec!["This is a sentence."];
//!     let embeddings = rubert.run(sentences)?;
//!
//!     Ok(())
//! }
//! ```

mod builder;
mod model;
mod pipeline;
mod pooler;
mod tokenizer;
mod utils;

pub use crate::{
    builder::{Builder, BuilderError},
    pipeline::{Embeddings, RuBert, RuBertError},
    pooler::Pooler,
};
pub(crate) use tract_onnx::prelude::tract_ndarray as ndarray;

/// Path to the current onnx model file.
#[cfg(test)]
const MODEL: &str = "../data/rubert_v0000/model.onnx";

/// Path to the current vocabulary file.
#[cfg(test)]
const VOCAB: &str = "../data/rubert_v0000/vocab.txt";
