use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    marker::PhantomData,
    path::Path,
};

use displaydoc::Display;
use thiserror::Error;

use crate::{model::BertModel, NonePooler};

#[derive(Debug, Display, Error)]
pub enum ConfigError {
    /// The token size must be greater than two to allow for special tokens
    TokenSize,
    /// Failed to load a data file: {0}
    DataFile(#[from] std::io::Error),
}

pub struct Config<'a, K, P> {
    pub(crate) model_kind: PhantomData<K>,
    pub(crate) vocab: Box<dyn BufRead + Send + 'a>,
    pub(crate) model: Box<dyn Read + Send + 'a>,
    pub(crate) accents: bool,
    pub(crate) lowercase: bool,
    pub(crate) token_size: usize,
    pub(crate) pooler: P,
}

impl<'a, K: BertModel> Config<'a, K, NonePooler> {
    pub fn from_readers(
        vocab: Box<dyn BufRead + Send + 'a>,
        model: Box<dyn Read + Send + 'a>,
    ) -> Self {
        Config {
            model_kind: Default::default(),
            vocab,
            model,
            accents: false,
            lowercase: true,
            token_size: 128,
            pooler: NonePooler,
        }
    }

    pub fn from_files(
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Self, ConfigError> {
        let vocab = Box::new(BufReader::new(File::open(vocab)?));
        let model = Box::new(BufReader::new(File::open(model)?));
        Ok(Self::from_readers(vocab, model))
    }
}

impl<'a, K: BertModel, P> Config<'a, K, P> {
    /// Whether the tokenizer keeps accents.
    ///
    /// Defaults to `false`.
    pub fn with_accents(mut self, accents: bool) -> Self {
        self.accents = accents;
        self
    }

    /// Whether the tokenizer lowercases.
    ///
    /// Defaults to `true`.
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Sets the token size for the tokenizer and the model.
    ///
    /// Defaults to [`K::TOKEN_RANGE`].
    ///
    /// # Errors
    /// Fails if `size` is less than two or greater than 512.
    pub fn with_token_size(mut self, size: usize) -> Result<Self, ConfigError> {
        if K::TOKEN_RANGE.contains(&size) {
            self.token_size = size;
            Ok(self)
        } else {
            Err(ConfigError::TokenSize)
        }
    }

    /// Sets pooling for the model.
    ///
    /// Defaults to `NonePooler`.
    pub fn with_pooling<NP>(self, pooler: NP) -> Config<'a, K, NP> {
        Config {
            vocab: self.vocab,
            model: self.model,
            model_kind: self.model_kind,
            accents: self.accents,
            lowercase: self.lowercase,
            token_size: self.token_size,
            pooler,
        }
    }
}
