use displaydoc::Display;
use thiserror::Error;

use crate::{
    model::{Model, ModelError},
    pooler::{Embedding1, Embedding2, PoolerError},
    tokenizer::{Tokenizer, TokenizerError},
    AveragePooler,
    FirstPooler,
    NonePooler,
};

/// A RuBert pipeline.
///
/// Can be created via the [`Builder`] and consists of a tokenizer, a model and a pooler.
///
/// [`Builder`]: crate::builder::Builder
pub struct RuBert<P> {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model: Model,
    pub(crate) pooler: P,
}

/// The potential errors of the [`RuBert`] pipeline.
#[derive(Debug, Display, Error)]
pub enum RuBertError {
    /// Failed to run the tokenizer: {0}
    Tokenizer(#[from] TokenizerError),
    /// Failed to run the model: {0}
    Model(#[from] ModelError),
    /// Failed to run the pooler: {0}
    Pooler(#[from] PoolerError),
}

impl RuBert<NonePooler> {
    /// Computes the embedding of the sequence.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<Embedding2, RuBertError> {
        let encoding = self.tokenizer.encode(sequence);
        let prediction = self.model.predict(encoding)?;
        self.pooler.pool(prediction).map_err(Into::into)
    }
}

impl RuBert<FirstPooler> {
    /// Computes the embedding of the sequence.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<Embedding1, RuBertError> {
        let encoding = self.tokenizer.encode(sequence);
        let prediction = self.model.predict(encoding)?;
        self.pooler.pool(prediction).map_err(Into::into)
    }
}

impl RuBert<AveragePooler> {
    /// Computes the embedding of the sequence.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<Embedding1, RuBertError> {
        let encoding = self.tokenizer.encode(sequence);
        let attention_mask = encoding.attention_mask.clone();
        let prediction = self.model.predict(encoding)?;
        self.pooler
            .pool(prediction, attention_mask)
            .map_err(Into::into)
    }
}

impl<P> RuBert<P> {
    /// Gets the token size.
    pub fn token_size(&self) -> usize {
        self.tokenizer.token_size
    }

    /// Gets the embedding size.
    pub fn embedding_size(&self) -> usize {
        self.model.embedding_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::Builder,
        pooler::{AveragePooler, FirstPooler, NonePooler},
        tests::{MODEL, VOCAB},
    };

    fn rubert<P>(pooler: P) -> RuBert<P> {
        Builder::from_files(VOCAB, MODEL)
            .unwrap()
            .with_accents(false)
            .with_lowercase(true)
            .with_token_size(64)
            .unwrap()
            .with_pooling(pooler)
            .build()
            .unwrap()
    }

    #[test]
    fn test_rubert_none() {
        let rubert = rubert(NonePooler);

        let embeddings = rubert.run("This is a sequence.").unwrap();
        assert_eq!(
            embeddings.shape(),
            &[rubert.token_size(), rubert.embedding_size()],
        );

        let embeddings = rubert.run("").unwrap();
        assert_eq!(
            embeddings.shape(),
            &[rubert.token_size(), rubert.embedding_size()],
        );
    }

    #[test]
    fn test_rubert_first() {
        let rubert = rubert(FirstPooler);

        let embeddings = rubert.run("This is a sequence.").unwrap();
        assert_eq!(embeddings.shape(), &[rubert.embedding_size()]);

        let embeddings = rubert.run("").unwrap();
        assert_eq!(embeddings.shape(), &[rubert.embedding_size()]);
    }

    #[test]
    fn test_rubert_average() {
        let rubert = rubert(AveragePooler);

        let embeddings = rubert.run("This is a sequence.").unwrap();
        assert_eq!(embeddings.shape(), &[rubert.embedding_size()]);

        let embeddings = rubert.run("").unwrap();
        assert_eq!(embeddings.shape(), &[rubert.embedding_size()]);
    }
}
