//! The KPE pipeline extracts key phrases from a sequence.
//!
//! ```no_run
//! use kpe::Builder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let kpe = Builder::from_files("vocab.txt", "bert.onnx", "cnn.onnx", "classifier.onnx")?
//!         .with_accents(false)
//!         .with_lowercase(true)
//!         .with_token_size(64)?
//!         .with_key_phrase_size(5)?
//!         .build()?;
//!
//!     let key_phrases = kpe.run("This is a sequence.")?;
//!     assert_eq!(key_phrases.len(), 12);
//!
//!     Ok(())
//! }
//! ```
#![cfg_attr(
    doc,
    forbid(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)
)]
#![forbid(unsafe_op_in_unsafe_fn)]

mod builder;
mod model;
mod pipeline;
mod tokenizer;

pub use crate::{
    builder::{Builder, BuilderError},
    pipeline::{Pipeline, PipelineError},
};

#[cfg(doc)]
pub use crate::{
    model::ModelError,
    tokenizer::{key_phrase::RankedKeyPhrases, TokenizerError},
};
