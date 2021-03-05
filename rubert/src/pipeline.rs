use std::cmp::min;

use displaydoc::Display;
use thiserror::Error;

use crate::{
    model::{Model, ModelError},
    pooler::{Embedding1, Embedding2, Embedding3, PoolerError},
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

    /// Computes the embeddings of the batch of sequences.
    ///
    /// The number of embeddings is the minimum between the number of sequences and the batch size.
    pub fn run_batch(&self, sequences: &[impl AsRef<str>]) -> Result<Embedding3, RuBertError> {
        let sequences = &sequences[..min(sequences.len(), self.batch_size())];
        let encodings = self.tokenizer.encode_batch(sequences);
        let predictions = self.model.predict(encodings)?;
        self.pooler
            .pool_batch(predictions, sequences.len())
            .map_err(Into::into)
    }
}

impl RuBert<FirstPooler> {
    /// Computes the embedding of the sequence.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<Embedding1, RuBertError> {
        let encoding = self.tokenizer.encode(sequence);
        let prediction = self.model.predict(encoding)?;
        self.pooler.pool(prediction).map_err(Into::into)
    }

    /// Computes the embeddings of the batch of sequences.
    ///
    /// The number of embeddings is the minimum between the number of sequences and the batch size.
    pub fn run_batch(&self, sequences: &[impl AsRef<str>]) -> Result<Embedding2, RuBertError> {
        let sequences = &sequences[..min(sequences.len(), self.batch_size())];
        let encodings = self.tokenizer.encode_batch(sequences);
        let predictions = self.model.predict(encodings)?;
        self.pooler
            .pool_batch(predictions, sequences.len())
            .map_err(Into::into)
    }
}

impl RuBert<AveragePooler> {
    /// Computes the embedding of the sequence.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<Embedding1, RuBertError> {
        let encoding = self.tokenizer.encode(sequence);
        let attention_mask = encoding.attention_masks.clone();
        let prediction = self.model.predict(encoding)?;
        self.pooler
            .pool(prediction, attention_mask)
            .map_err(Into::into)
    }

    /// Computes the embeddings of the batch of sequences.
    ///
    /// The number of embeddings is the minimum between the number of sequences and the batch size.
    pub fn run_batch(&self, sequences: &[impl AsRef<str>]) -> Result<Embedding2, RuBertError> {
        let sequences = &sequences[..min(sequences.len(), self.batch_size())];
        let encodings = self.tokenizer.encode_batch(sequences);
        let attention_masks = encodings.attention_masks.clone();
        let predictions = self.model.predict(encodings)?;
        self.pooler
            .pool_batch(predictions, attention_masks, sequences.len())
            .map_err(Into::into)
    }
}

impl<P> RuBert<P> {
    /// Gets the batch size.
    pub fn batch_size(&self) -> usize {
        self.model.batch_size()
    }

    /// Gets the token size.
    pub fn token_size(&self) -> usize {
        self.model.token_size()
    }

    /// Gets the embedding size.
    pub fn embedding_size(&self) -> usize {
        self.model.embedding_size()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::Builder,
        pooler::{AveragePooler, FirstPooler, NonePooler},
        RuBert,
        MODEL,
        VOCAB,
    };

    fn rubert<P>(pooler: P) -> RuBert<P> {
        Builder::from_files(VOCAB, MODEL)
            .unwrap()
            .with_accents(true)
            .with_lowercase(true)
            .with_batch_size(5)
            .unwrap()
            .with_token_size(64)
            .unwrap()
            .with_pooling(pooler)
            .build()
            .unwrap()
    }

    #[test]
    fn test_rubert_none() {
        let rubert = rubert(NonePooler);

        let embeddings = rubert.run("This is a sentence.").unwrap();
        assert_eq!(
            embeddings.shape(),
            &[rubert.token_size(), rubert.embedding_size()],
        );

        let embeddings = rubert
            .run_batch(&["bank vault", "bank robber", "river bank"])
            .unwrap();
        assert_eq!(
            embeddings.shape(),
            &[3, rubert.token_size(), rubert.embedding_size()],
        );

        let embeddings = rubert.run_batch(&[""; 0]).unwrap();
        assert_eq!(
            embeddings.shape(),
            &[0, rubert.token_size(), rubert.embedding_size()]
        );

        let embeddings = rubert.run_batch(&[""; 10]).unwrap();
        assert_eq!(
            embeddings.shape(),
            &[5, rubert.token_size(), rubert.embedding_size()]
        );
    }

    #[test]
    fn test_rubert_first() {
        let rubert = rubert(FirstPooler);

        let embeddings = rubert.run("This is a sentence.").unwrap();
        assert_eq!(embeddings.shape(), &[rubert.embedding_size()]);

        let embeddings = rubert
            .run_batch(&["bank vault", "bank robber", "river bank"])
            .unwrap();
        assert_eq!(embeddings.shape(), &[3, rubert.embedding_size()]);

        let embeddings = rubert.run_batch(&[""; 0]).unwrap();
        assert_eq!(embeddings.shape(), &[0, rubert.embedding_size()]);

        let embeddings = rubert.run_batch(&[""; 10]).unwrap();
        assert_eq!(embeddings.shape(), &[5, rubert.embedding_size()]);
    }

    #[test]
    fn test_rubert_average() {
        let rubert = rubert(AveragePooler);

        let embeddings = rubert.run("This is a sentence.").unwrap();
        assert_eq!(embeddings.shape(), &[rubert.embedding_size()]);

        let embeddings = rubert
            .run_batch(&["bank vault", "bank robber", "river bank"])
            .unwrap();
        assert_eq!(embeddings.shape(), &[3, rubert.embedding_size()]);

        let embeddings = rubert.run_batch(&[""; 0]).unwrap();
        assert_eq!(embeddings.shape(), &[0, rubert.embedding_size()]);

        let embeddings = rubert.run_batch(&[""; 10]).unwrap();
        assert_eq!(embeddings.shape(), &[5, rubert.embedding_size()]);
    }
}
