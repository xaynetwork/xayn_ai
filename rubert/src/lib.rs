#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]
//! The RuBert pipeline computes embeddings of sequences.
//!
//! Sequences are anything string-like and can also be single words or snippets. The embeddings are
//! f32-arrays and their shape depends on the pooling strategy.
//!
//! ```no_run
//! use rubert::{SMBertBuilder, FirstPooler};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let rubert = SMBertBuilder::from_files("vocab.txt", "model.onnx")?
//!         .with_accents(false)
//!         .with_lowercase(true)
//!         .with_token_size(64)?
//!         .with_pooling(FirstPooler)
//!         .build()?;
//!
//!     let embedding = rubert.run("This is a sequence.")?;
//!     assert_eq!(embedding.shape(), [rubert.embedding_size()]);
//!
//!     Ok(())
//! }
//! ```

mod builder;
mod model;
mod pipeline;
mod pooler;
mod tokenizer;

use crate::model::kinds;

pub use crate::{
    builder::{Builder, BuilderError},
    pipeline::{Pipeline, PipelineError},
    pooler::{AveragePooler, Embedding1, Embedding2, FirstPooler, NonePooler},
};

#[allow(clippy::upper_case_acronyms)]
pub type SMBert = Pipeline<kinds::SMBert, AveragePooler>;
#[allow(clippy::upper_case_acronyms)]
pub type QAMBert = Pipeline<kinds::QAMBert, AveragePooler>;

#[allow(clippy::upper_case_acronyms)]
pub type SMBertBuilder<V, M> = Builder<V, M, kinds::SMBert, NonePooler>;
#[allow(clippy::upper_case_acronyms)]
pub type QAMBertBuilder<V, M> = Builder<V, M, kinds::QAMBert, NonePooler>;

#[cfg(doc)]
pub use crate::{
    model::ModelError,
    pooler::{Embedding, PoolerError},
    tokenizer::TokenizerError,
};

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use std::{
        collections::hash_map::DefaultHasher,
        fs::File,
        hash::{Hash, Hasher},
        io::{BufReader, Read},
    };

    /// Path to the current vocabulary file.
    /// The vocabulary is in common between the bert models
    pub const VOCAB: &str = "../data/rubert_v0001/vocab.txt";

    /// Path to the current onnx smbert model file.
    pub const SMBERT_MODEL: &str = "../data/rubert_v0001/smbert.onnx";

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn hash_file(file: &str) -> u64 {
        let mut reader = BufReader::new(File::open(file).unwrap());
        let mut buffer = vec![0; 1024];
        let mut hasher = DefaultHasher::new();
        loop {
            if reader.read(&mut buffer).unwrap() == 0 {
                break;
            }
            buffer.hash(&mut hasher);
        }
        hasher.finish()
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_vocab_unchanged() {
        assert_eq!(hash_file(VOCAB), 5045043227147541355);
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_model_unchanged() {
        assert_eq!(hash_file(SMBERT_MODEL), 13727150546539837987);
    }
}
