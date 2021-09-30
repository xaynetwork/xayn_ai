use std::{
    fs::File,
    io::{BufRead, BufReader, Error as IoError, Read},
    path::Path,
};

use displaydoc::Display;
use thiserror::Error;

use crate::{
    model::{
        bert::{BertModel, BertModelError},
        classifier::{ClassifierModel, ClassifierModelError},
        cnn::{CnnModel, CnnModelError},
    },
    pipeline::Pipeline,
    tokenizer::{Tokenizer, TokenizerError},
};

/// A builder to create a [`Pipeline`].
pub struct Builder<V, M> {
    vocab: V,
    bert: M,
    cnn: M,
    classifier: M,
    accents: bool,
    lowercase: bool,
    token_size: usize,
    key_phrase_size: usize,
}

/// The potential errors of the builder.
#[derive(Debug, Display, Error)]
pub enum BuilderError {
    /// The token size must be at least two to allow for special tokens
    TokenSize,
    /// The maximum key phrase words must be at least one
    MaxKeyPhraseWords,
    /// Failed to load a data file: {0}
    DataFile(#[from] IoError),
    /// Failed to build the tokenizer: {0}
    Tokenizer(#[from] TokenizerError),
    /// Failed to build the Bert model: {0}
    BertModel(#[from] BertModelError),
    /// Failed to build the CNN model: {0}
    CnnModel(#[from] CnnModelError),
    /// Failed to build the Classifier model: {0}
    ClassifierModel(#[from] ClassifierModelError),
}

impl Builder<BufReader<File>, BufReader<File>> {
    /// Creates a [`Pipeline`] builder from a vocabulary and model files.
    pub fn from_files(
        vocab: impl AsRef<Path>,
        bert: impl AsRef<Path>,
        cnn: impl AsRef<Path>,
        classifier: impl AsRef<Path>,
    ) -> Result<Self, BuilderError> {
        let vocab = BufReader::new(File::open(vocab)?);
        let bert = BufReader::new(File::open(bert)?);
        let cnn = BufReader::new(File::open(cnn)?);
        let classifier = BufReader::new(File::open(classifier)?);
        Ok(Self::new(vocab, bert, cnn, classifier))
    }
}

impl<V, M> Builder<V, M> {
    /// Creates a [`Pipeline`] builder from an in-memory vocabulary and models.
    pub fn new(vocab: V, bert: M, cnn: M, classifier: M) -> Self {
        Self {
            vocab,
            bert,
            cnn,
            classifier,
            accents: false,
            lowercase: true,
            token_size: 1024,
            key_phrase_size: 5,
        }
    }

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

    /// Sets the token size for the tokenizer and the models.
    ///
    /// Defaults to `1024`.
    ///
    /// # Errors
    /// Fails if `size` is less than two.
    pub fn with_token_size(mut self, size: usize) -> Result<Self, BuilderError> {
        if size > 1 {
            self.token_size = size;
            Ok(self)
        } else {
            Err(BuilderError::TokenSize)
        }
    }

    /// Sets the maximum key phrase words for the tokenizer and the models.
    ///
    /// Defaults to `5`.
    ///
    /// # Errors
    /// Fails if `words` is less than one.
    pub fn with_key_phrase_size(mut self, words: usize) -> Result<Self, BuilderError> {
        if words > 0 {
            self.key_phrase_size = words;
            Ok(self)
        } else {
            Err(BuilderError::MaxKeyPhraseWords)
        }
    }

    /// Builds a [`Pipeline`].
    ///
    /// # Errors
    /// Fails on invalid tokenizer or model settings.
    pub fn build(self) -> Result<Pipeline, BuilderError>
    where
        V: BufRead,
        M: Read,
    {
        let tokenizer = Tokenizer::new(
            self.vocab,
            self.accents,
            self.lowercase,
            self.token_size,
            self.key_phrase_size,
        )?;
        let bert = BertModel::new(self.bert, self.token_size)?;
        let cnn = CnnModel::new(self.cnn, self.token_size, bert.embedding_size)?;
        let classifier =
            ClassifierModel::new(self.classifier, self.key_phrase_size, cnn.out_channel_size)?;

        Ok(Pipeline {
            tokenizer,
            bert,
            cnn,
            classifier,
        })
    }
}
