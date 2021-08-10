#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]
//! The RuBert pipeline computes embeddings of sequences.
//!
//! Sequences are anything string-like and can also be single words or snippets. The embeddings are
//! f32-arrays and their shape depends on the pooling strategy.
//!
//! ```no_run
//! use rubert::{FirstPooler, SMBertBuilder};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let smbert = SMBertBuilder::from_files("vocab.txt", "model.onnx")?
//!         .with_accents(false)
//!         .with_lowercase(true)
//!         .with_token_size(64)?
//!         .with_pooling(FirstPooler)
//!         .build()?;
//!
//!     let embedding = smbert.run("This is a sequence.")?;
//!     assert_eq!(embedding.shape(), [smbert.embedding_size()]);
//!
//!     Ok(())
//! }
//! ```

mod builder;
mod model;
mod pipeline;
mod pooler;
mod tokenizer;

pub use crate::{
    builder::{Builder, BuilderError},
    model::kinds,
    pipeline::{Pipeline, PipelineError},
    pooler::{AveragePooler, Embedding1, Embedding2, FirstPooler, NonePooler},
};

/// A sentence (embedding) multilingual Bert pipeline.
#[allow(clippy::upper_case_acronyms)]
pub type SMBert = Pipeline<kinds::SMBert, AveragePooler>;

/// A question answering (embedding) multilingual Bert pipeline.
#[allow(clippy::upper_case_acronyms)]
pub type QAMBert = Pipeline<kinds::QAMBert, AveragePooler>;

/// A builder to create a [`SMBert`] pipeline.
#[allow(clippy::upper_case_acronyms)]
pub type SMBertBuilder<V, M> = Builder<V, M, kinds::SMBert, NonePooler>;

/// A builder to create a [`QAMBert`] pipeline.
#[allow(clippy::upper_case_acronyms)]
pub type QAMBertBuilder<V, M> = Builder<V, M, kinds::QAMBert, NonePooler>;

#[cfg(doc)]
pub use crate::{
    model::ModelError,
    pooler::{Embedding, PoolerError},
    tokenizer::TokenizerError,
};
