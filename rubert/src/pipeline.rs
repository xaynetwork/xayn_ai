use displaydoc::Display;
use thiserror::Error;

use crate::{
    model::{BertModel, Model, ModelError},
    pooler::{Embedding1, Embedding2, PoolerError},
    tokenizer::{Tokenizer, TokenizerError},
    AveragePooler,
    Configuration,
    FirstPooler,
    NonePooler,
};

/// A pipeline for a bert model.
///
/// Can be created via the [`Builder`] and consists of a tokenizer, a model and a pooler.
///
/// [`Builder`]: crate::builder::Builder
pub struct Pipeline<K, P> {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model: Model<K>,
    pub(crate) pooler: P,
}

/// The potential errors of the [`Pipeline`].
#[derive(Debug, Display, Error)]
pub enum PipelineError {
    /// Failed to run the tokenizer: {0}
    Tokenizer(#[from] TokenizerError),
    /// Failed to run the model: {0}
    Model(#[from] ModelError),
    /// Failed to run the pooler: {0}
    Pooler(#[from] PoolerError),
    /// Failed to build the tokenizer: {0}
    TokenizerBuild(#[source] TokenizerError),
    /// Failed to build the model: {0}
    ModelBuild(#[source] ModelError),
}

impl<K, P> Pipeline<K, P>
where
    K: BertModel,
{
    pub fn from(config: Configuration<K, P>) -> Result<Self, PipelineError> {
        let tokenizer = Tokenizer::new(
            config.vocab,
            config.accents,
            config.lowercase,
            config.token_size,
        )
        .map_err(PipelineError::TokenizerBuild)?;

        let model =
            Model::new(config.model, config.token_size).map_err(PipelineError::ModelBuild)?;

        Ok(Pipeline {
            tokenizer,
            model,
            pooler: config.pooler,
        })
    }
}

impl<K> Pipeline<K, NonePooler>
where
    K: BertModel,
{
    /// Computes the embedding of the sequence.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<Embedding2, PipelineError> {
        let encoding = self.tokenizer.encode(sequence);
        let prediction = self.model.predict(encoding)?;
        self.pooler.pool(prediction).map_err(Into::into)
    }
}

impl<K> Pipeline<K, FirstPooler>
where
    K: BertModel,
{
    /// Computes the embedding of the sequence.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<Embedding1, PipelineError> {
        let encoding = self.tokenizer.encode(sequence);
        let prediction = self.model.predict(encoding)?;
        self.pooler.pool(prediction).map_err(Into::into)
    }
}

impl<K> Pipeline<K, AveragePooler>
where
    K: BertModel,
{
    /// Computes the embedding of the sequence.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<Embedding1, PipelineError> {
        let encoding = self.tokenizer.encode(sequence);
        let attention_mask = encoding.attention_mask.clone();
        let prediction = self.model.predict(encoding)?;
        self.pooler
            .pool(prediction, attention_mask)
            .map_err(Into::into)
    }
}

impl<K, P> Pipeline<K, P>
where
    K: BertModel,
{
    /// Gets the token size.
    pub fn token_size(&self) -> usize {
        self.model.token_size
    }

    /// Gets the embedding size.
    pub fn embedding_size(&self) -> usize {
        K::EMBEDDING_SIZE
    }
}

#[cfg(test)]
mod tests {
    use test_utils::smbert::{model, vocab};

    use super::*;
    use crate::{
        model::kinds::SMBert,
        pooler::{AveragePooler, FirstPooler, NonePooler},
    };

    fn pipeline<P>(pooler: P) -> Pipeline<SMBert, P> {
        let config = Configuration::from_files(vocab().unwrap(), model().unwrap())
            .unwrap()
            .with_pooling(pooler);

        Pipeline::from(config).unwrap()
    }

    #[test]
    fn test_pipeline_none() {
        let pipeline = pipeline(NonePooler);

        let embeddings = pipeline.run("This is a sequence.").unwrap();
        assert_eq!(
            embeddings.shape(),
            &[pipeline.token_size(), pipeline.embedding_size()],
        );

        let embeddings = pipeline.run("").unwrap();
        assert_eq!(
            embeddings.shape(),
            &[pipeline.token_size(), pipeline.embedding_size()],
        );
    }

    #[test]
    fn test_pipeline_first() {
        let pipeline = pipeline(FirstPooler);

        let embeddings = pipeline.run("This is a sequence.").unwrap();
        assert_eq!(embeddings.shape(), &[pipeline.embedding_size()]);

        let embeddings = pipeline.run("").unwrap();
        assert_eq!(embeddings.shape(), &[pipeline.embedding_size()]);
    }

    #[test]
    fn test_pipeline_average() {
        let pipeline = pipeline(AveragePooler);

        let embeddings = pipeline.run("This is a sequence.").unwrap();
        assert_eq!(embeddings.shape(), &[pipeline.embedding_size()]);

        let embeddings = pipeline.run("").unwrap();
        assert_eq!(embeddings.shape(), &[pipeline.embedding_size()]);
    }
}
