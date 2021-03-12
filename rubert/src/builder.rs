use std::{
    fs::File,
    io::{BufRead, BufReader, Error as IoError, Read},
    path::Path,
};

use displaydoc::Display;
use thiserror::Error;

use crate::{
    model::{Model, ModelError},
    pipeline::RuBert,
    pooler::NonePooler,
    tokenizer::{Tokenizer, TokenizerError},
};

/// A builder to create a [`RuBert`] pipeline.
pub struct Builder<V, M, P> {
    vocab: V,
    model: M,
    accents: bool,
    lowercase: bool,
    batch_size: usize,
    token_size: usize,
    pooler: P,
}

/// The potential errors of the builder.
#[derive(Debug, Display, Error)]
pub enum BuilderError {
    /// The batch size must be greater than zero
    BatchSize,
    /// The token size must be greater than two to allow for special tokens
    TokenSize,
    /// Failed to load a data file: {0}
    DataFile(#[from] IoError),
    /// Failed to build the tokenizer: {0}
    Tokenizer(#[from] TokenizerError),
    /// Failed to build the model: {0}
    Model(#[from] ModelError),
}

impl<V, M> Builder<V, M, NonePooler>
where
    V: BufRead,
    M: Read,
{
    /// Creates a [`RuBert`] pipeline builder from an in-memory vocabulary and model.
    pub fn new(vocab: V, model: M) -> Self {
        Self {
            vocab,
            model,
            accents: true,
            lowercase: true,
            batch_size: 10,
            token_size: 128,
            pooler: NonePooler,
        }
    }
}

impl Builder<BufReader<File>, BufReader<File>, NonePooler> {
    /// Creates a [`RuBert`] pipeline builder from a vocabulary and model file.
    pub fn from_files(
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Self, BuilderError> {
        let vocab = BufReader::new(File::open(vocab)?);
        let model = BufReader::new(File::open(model)?);
        Ok(Self::new(vocab, model))
    }
}

impl<V, M, P> Builder<V, M, P> {
    /// Toggles accent stripping for the tokenizer.
    ///
    /// Defaults to `true`.
    pub fn with_accents(mut self, toggle: bool) -> Self {
        self.accents = toggle;
        self
    }

    /// Toggles lower casing for the tokenizer.
    ///
    /// Defaults to `true`.
    pub fn with_lowercase(mut self, toggle: bool) -> Self {
        self.lowercase = toggle;
        self
    }

    /// Sets the batch size for the model.
    ///
    /// Defaults to `10`.
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
    /// Defaults to `128`.
    ///
    /// # Errors
    /// Fails if `size` is less than two.
    pub fn with_token_size(mut self, size: usize) -> Result<Self, BuilderError> {
        if size < 2 {
            Err(BuilderError::TokenSize)
        } else {
            self.token_size = size;
            Ok(self)
        }
    }

    /// Sets pooling for the model.
    ///
    /// Defaults to `None`.
    pub fn with_pooling<Q>(self, pooler: Q) -> Builder<V, M, Q> {
        Builder {
            vocab: self.vocab,
            model: self.model,
            accents: self.accents,
            lowercase: self.lowercase,
            batch_size: self.batch_size,
            token_size: self.token_size,
            pooler,
        }
    }

    /// Builds a [`RuBert`] pipeline.
    ///
    /// # Errors
    /// Fails on invalid tokenizer or model settings.
    pub fn build(self) -> Result<RuBert<P>, BuilderError>
    where
        V: BufRead,
        M: Read,
    {
        let tokenizer = Tokenizer::new(
            self.vocab,
            self.accents,
            self.lowercase,
            self.batch_size,
            self.token_size,
        )?;
        let model = Model::new(self.model, self.batch_size, self.token_size)?;

        Ok(RuBert {
            tokenizer,
            model,
            pooler: self.pooler,
        })
    }
}
