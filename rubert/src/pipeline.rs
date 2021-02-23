use derive_more::Deref;
use displaydoc::Display;
use thiserror::Error;
use tokenizers::tokenizer::EncodeInput;

use crate::{
    model::{Model, ModelError},
    pooler::{Pooler, Poolings},
    tokenizer::{Tokenizer, TokenizerError},
    utils::ArcArrayD,
};

/// A RuBert pipeline.
///
/// Can be created via the [`Builder`] and consists of a tokenizer, a model and optionally a pooler.
///
/// [`Builder`]: crate::builder::Builder
pub struct RuBert {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model: Model,
    pub(crate) pooler: Pooler,
}

/// Potential errors of the [`RuBert`] pipeline.
#[derive(Debug, Display, Error)]
pub enum RuBertError {
    /// Failed to run the tokenizer: {0}.
    Tokenizer(#[from] TokenizerError),
    /// Failed to run the model: {0}.
    Model(#[from] ModelError),
}

/// The embeddings of the sentences.
#[derive(Clone, Deref)]
pub struct Embeddings(pub(crate) ArcArrayD<f32>);

impl From<Poolings> for Embeddings {
    fn from(poolings: Poolings) -> Self {
        match poolings {
            Poolings::None(pooled) => Embeddings(pooled.into_dyn()),
            Poolings::First(pooled) => Embeddings(pooled.into_dyn()),
            Poolings::Average(pooled) => Embeddings(pooled.into_dyn()),
        }
    }
}

impl RuBert {
    /// Runs the pipeline to compute embeddings of the sentences.
    ///
    /// The embeddings are computed from the sentences by tokenization, prediction and optional
    /// pooling.
    ///
    /// # Errors
    /// - The tokenization fails if any of the normalization, tokenization, pre- or post-processing
    /// steps fail.
    /// - The prediction fails on dimensionality mismatches for the tokenized sentences regarding
    /// the loaded onnx model.
    pub fn run<'s>(
        &self,
        sentences: Vec<impl Into<EncodeInput<'s>> + Send>,
    ) -> Result<Embeddings, RuBertError> {
        let encodings = self.tokenizer.encode(sentences)?;
        let attention_masks = encodings.attention_masks.clone();
        let predictions = self.model.predict(encodings)?;
        let poolings = self.pooler.pool(predictions, attention_masks);

        Ok(poolings.into())
    }

    /// Returns the batch size of the model pipeline.
    pub fn batch_size(&self) -> usize {
        self.model.batch_size()
    }

    /// Returns the token size of the model pipeline.
    pub fn token_size(&self) -> usize {
        self.model.token_size()
    }

    /// Returns the embedding size of the model pipeline.
    pub fn embedding_size(&self) -> usize {
        self.model.embedding_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{builder::Builder, MODEL, VOCAB};

    #[test]
    fn test_pipeline() {
        let rubert = Builder::from_files(VOCAB, MODEL)
            .unwrap()
            .with_strip_accents(true)
            .with_lowercase(true)
            .with_batch_size(10)
            .unwrap()
            .with_token_size(64)
            .unwrap()
            .with_pooling(Pooler::First)
            .build()
            .unwrap();

        let sentences = vec!["This is a sentence."];
        let embeddings = rubert.run(sentences).unwrap();
        assert_eq!(embeddings.shape(), &[1, rubert.embedding_size()]);

        let sentences = vec!["bank vault", "bank robber", "river bank"];
        let embeddings = rubert.run(sentences).unwrap();
        assert_eq!(embeddings.shape(), &[3, rubert.embedding_size()]);
    }
}
