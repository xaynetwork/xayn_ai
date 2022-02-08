use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use displaydoc::Display;
use thiserror::Error;

use crate::model::bert::Bert;

#[derive(Debug, Display, Error)]
pub enum ConfigError {
    /// The token size must be at least two to allow for special tokens
    TokenSize,
    /// The maximum number of returned key phrases must be at least one if given
    KeyPhraseMaxCount,
    /// The minimum score of returned key phrases must be finite if given
    KeyPhraseMinScore,
    /// Failed to load a data file: {0}
    DataFile(#[from] std::io::Error),
}

pub struct Config<'a> {
    pub(crate) vocab: Box<dyn BufRead + Send + 'a>,
    pub(crate) model: Box<dyn Read + Send + 'a>,
    pub(crate) cnn: Box<dyn Read + Send + 'a>,
    pub(crate) classifier: Box<dyn Read + Send + 'a>,
    pub(crate) accents: bool,
    pub(crate) lowercase: bool,
    pub(crate) token_size: usize,
    pub(crate) key_phrase_max_count: Option<usize>,
    pub(crate) key_phrase_min_score: Option<f32>,
}

impl<'a> Config<'a> {
    pub fn from_readers(
        vocab: Box<dyn BufRead + Send + 'a>,
        model: Box<dyn Read + Send + 'a>,
        cnn: Box<dyn Read + Send + 'a>,
        classifier: Box<dyn Read + Send + 'a>,
    ) -> Self {
        Config {
            vocab,
            model,
            cnn,
            classifier,
            accents: false,
            lowercase: true,
            token_size: *Bert::TOKEN_RANGE.end(),
            key_phrase_max_count: None,
            key_phrase_min_score: None,
        }
    }

    pub fn from_files(
        vocab: impl AsRef<Path>,
        model: impl AsRef<Path>,
        cnn: impl AsRef<Path>,
        classifier: impl AsRef<Path>,
    ) -> Result<Self, ConfigError> {
        let vocab = Box::new(BufReader::new(File::open(vocab)?));
        let model = Box::new(BufReader::new(File::open(model)?));
        let cnn = Box::new(BufReader::new(File::open(cnn)?));
        let classifier = Box::new(BufReader::new(File::open(classifier)?));
        Ok(Self::from_readers(vocab, model, cnn, classifier))
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
    /// Defaults to [`Bert::TOKEN_RANGE.max`].
    ///
    /// # Errors
    /// Fails if `size` is less than two or greater than 512.
    pub fn with_token_size(mut self, size: usize) -> Result<Self, ConfigError> {
        if Bert::TOKEN_RANGE.contains(&size) {
            self.token_size = size;
            Ok(self)
        } else {
            Err(ConfigError::TokenSize)
        }
    }

    /// Sets the optional maximum number of returned ranked key phrases.
    ///
    /// Defaults to `None`. The actual returned number of ranked key phrases might be less than the
    /// count depending on the lower threshold for the key phrase ranking scores.
    ///
    /// # Errors
    /// Fails if `count` is given and less than one.
    pub fn with_key_phrase_max_count(mut self, count: Option<usize>) -> Result<Self, ConfigError> {
        if count.is_none() || count > Some(0) {
            self.key_phrase_max_count = count;
            Ok(self)
        } else {
            Err(ConfigError::KeyPhraseMaxCount)
        }
    }

    /// Sets the optional lower threshold for scores of returned ranked key phrases.
    ///
    /// Defaults to `None`. The actual returned number of ranked key phrases might be less than
    /// indicated by the threshold depending on the upper count for the key phrases.
    ///
    /// # Errors
    /// Fails if `score` is given and not finite.
    pub fn with_key_phrase_min_score(mut self, score: Option<f32>) -> Result<Self, ConfigError> {
        if score.is_none() || score.map(f32::is_finite).unwrap_or_default() {
            self.key_phrase_min_score = score;
            Ok(self)
        } else {
            Err(ConfigError::KeyPhraseMinScore)
        }
    }
}
