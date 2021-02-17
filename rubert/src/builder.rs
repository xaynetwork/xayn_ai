use std::path::Path;

use displaydoc::Display;
use thiserror::Error;

use crate::{
    model::{Model, ModelError},
    pipeline::RuBert,
    pooler::Pooler,
    tokenizer::{Tokenizer, TokenizerError},
};

/// A builder to create a [`RuBert`] pipeline.
pub struct Builder<V, M> {
    vocab: V,
    model: M,
    strip_accents: bool,
    lowercase: bool,
    batch_size: usize,
    token_size: usize,
    pooler: Pooler,
}

/// Potential errors of the [`RuBert`] [`Builder`].
#[derive(Debug, Display, Error)]
pub enum BuilderError {
    /// The batch size must be greater than zero.
    BatchSize,
    /// The token size must be greater than two to allow for special tokens.
    TokenSize,
    /// Failed to build the tokenizer: {0}.
    Tokenizer(#[from] TokenizerError),
    /// Failed to build the model: {0}.
    Model(#[from] ModelError),
}

impl<V, M> Builder<V, M>
where
    V: AsRef<Path>,
    M: AsRef<Path>,
{
    /// Creates a [`RuBert`] pipeline builder.
    ///
    /// The default settings are:
    /// - Strips accents and makes lower case.
    /// - Supports batch size of 10 and token size of 128.
    /// - Applies no additional pooling.
    pub fn new(vocab: V, model: M) -> Self {
        Self {
            vocab,
            model,
            strip_accents: true,
            lowercase: true,
            batch_size: 10,
            token_size: 128,
            pooler: Pooler::None,
        }
    }

    /// Toggles accent stripping for the tokenizer.
    pub fn with_strip_accents(mut self, toggle: bool) -> Self {
        self.strip_accents = toggle;
        self
    }

    /// Toggles lower casing for the tokenizer.
    pub fn with_lowercase(mut self, toggle: bool) -> Self {
        self.lowercase = toggle;
        self
    }

    /// Sets the batch size for the model.
    ///
    /// # Errors
    /// Fails if `size` is zero.
    pub fn with_batch_size(mut self, size: usize) -> Result<Self, BuilderError> {
        if size == 0 {
            Err(BuilderError::BatchSize)
        } else {
            self.batch_size = size;
            Ok(self)
        }
    }

    /// Sets the token size for the tokenizer and the model.
    ///
    /// # Errors
    /// Fails if `size` is less than two.
    pub fn with_token_size(mut self, size: usize) -> Result<Self, BuilderError> {
        if self.token_size < 2 {
            return Err(BuilderError::TokenSize);
        } else {
            self.token_size = size;
            Ok(self)
        }
    }

    /// Sets pooling for the model.
    pub fn with_pooling(mut self, pooler: Pooler) -> Self {
        self.pooler = pooler;
        self
    }

    /// Builds a [`RuBert`] pipeline.
    ///
    /// # Errors
    /// Fails on invalid tokenizer or model settings.
    pub fn build(self) -> Result<RuBert, BuilderError> {
        let tokenizer = Tokenizer::new(
            self.vocab,
            self.strip_accents,
            self.lowercase,
            self.token_size,
        )?;
        let model = Model::new(self.model, self.batch_size, self.token_size)?;
        let pooler = self.pooler;

        Ok(RuBert {
            tokenizer,
            model,
            pooler,
        })
    }
}
