use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use displaydoc::Display;
use num_traits::FromPrimitive;
use thiserror::Error;

use crate::{
    model::{Model, ModelError, Vocab},
    normalizer::Normalizer,
    post_tokenizer::{
        padding::{Padding, PaddingError},
        truncation::{Truncation, TruncationError},
        PostTokenizer,
        PostTokenizerError,
    },
    pre_tokenizer::PreTokenizer,
    tokenizer::Tokenizer,
    SmallString,
};

/// A builder to create a Bert tokenizer.
pub struct Builder<N> {
    // normalizer
    cleanup: bool,
    chinese: bool,
    accents: bool,
    lowercase: bool,
    // model
    vocab: Vocab<N>,
    unk: SmallString,
    prefix: SmallString,
    max_chars: usize,
    // post-tokenizer
    cls: SmallString,
    sep: SmallString,
    trunc: Truncation,
    pad: Padding<N>,
}

/// The potential errors of the builder.
#[derive(Debug, Display, Error, PartialEq)]
pub enum BuilderError {
    /// Failed to build the model: {0}
    Model(#[from] ModelError),
    /// Failed to build the post-tokenizer: {0}
    PostTokenizer(#[from] PostTokenizerError),
    /// Failed to build the truncation strategy: {0}
    Truncation(#[from] TruncationError),
    /// Failed to build the padding strategy: {0}
    Padding(#[from] PaddingError),
}

impl<N> Builder<N> {
    /// Creates a [`Tokenizer`] builder from an in-memory vocabulary.
    ///
    /// # Errors
    /// Fails on invalid vocabularies.
    pub fn new(vocab: impl BufRead) -> Result<Self, BuilderError>
    where
        N: FromPrimitive,
    {
        Ok(Self {
            // normalizer
            cleanup: true,
            chinese: true,
            accents: false,
            lowercase: true,
            // model
            vocab: Model::parse_vocab(vocab)?,
            unk: "[UNK]".into(),
            prefix: "##".into(),
            max_chars: 100,
            // post-tokenizer
            cls: "[CLS]".into(),
            sep: "[SEP]".into(),
            trunc: Truncation::none(),
            pad: Padding::none(),
        })
    }

    /// Creates a [`Tokenizer`] builder from a vocabulary file.
    ///
    /// # Errors
    /// Fails on invalid vocabularies.
    pub fn from_file(vocab: impl AsRef<Path>) -> Result<Self, BuilderError>
    where
        N: FromPrimitive,
    {
        Self::new(BufReader::new(File::open(vocab).map_err(ModelError::from)?))
    }

    /// Configures the normalizer.
    ///
    /// Configurable by:
    /// - Cleans any control characters and replaces all sorts of whitespace by ` `. Defaults to
    /// `true`.
    /// - Separates Chinese characters by whitespace so they get split. Defaults to `true`.
    /// - Keeps accents of characters. Defaults to `false`.
    /// - Lowercases characters. Defaults to `true`.
    pub fn with_normalizer(
        mut self,
        cleanup: bool,
        chinese: bool,
        accents: bool,
        lowercase: bool,
    ) -> Self {
        self.cleanup = cleanup;
        self.chinese = chinese;
        self.accents = accents;
        self.lowercase = lowercase;
        self
    }

    /// Configures the model.
    ///
    /// Configurable by:
    /// - The unknown token. Defaults to `[UNK]`.
    /// - The continuing subword prefix. Defaults to `##`.
    /// - The maximum number of characters per word. Defaults to `100`.
    pub fn with_model(
        mut self,
        unk: impl AsRef<str>,
        prefix: impl AsRef<str>,
        max_chars: usize,
    ) -> Self {
        self.unk = unk.as_ref().into();
        self.prefix = prefix.as_ref().into();
        self.max_chars = max_chars;
        self
    }

    /// Configures the post-tokenizer.
    ///
    /// Configurable by:
    /// - The class token. Defaults to `"[CLS]"`.
    /// - The separation token. Defaults to `"[SEP]"`.
    pub fn with_post_tokenizer(mut self, cls: impl AsRef<str>, sep: impl AsRef<str>) -> Self {
        self.cls = cls.as_ref().into();
        self.sep = sep.as_ref().into();
        self
    }

    /// Configures the truncation strategy.
    ///
    /// Defaults to no truncation.
    pub fn with_truncation(mut self, trunc: Truncation) -> Self {
        self.trunc = trunc;
        self
    }

    /// Configures the padding strategy.
    ///
    /// Defaults to no padding.
    pub fn with_padding(mut self, pad: Padding<N>) -> Self {
        self.pad = pad;
        self
    }

    /// Builds the tokenizer.
    ///
    /// # Errors
    /// Fails on invalid configurations.
    pub fn build(self) -> Result<Tokenizer<N>, BuilderError>
    where
        N: Copy,
    {
        let normalizer = Normalizer::new(self.cleanup, self.chinese, self.accents, self.lowercase);
        let pre_tokenizer = PreTokenizer;
        let model = Model::new(self.vocab, self.unk, self.prefix, self.max_chars)?;
        let post_tokenizer = PostTokenizer::new(self.cls, self.sep, &model.vocab)?;
        let truncation = self.trunc.validate()?;
        let padding = self.pad.validate(&model.vocab)?;

        Ok(Tokenizer {
            normalizer,
            pre_tokenizer,
            model,
            post_tokenizer,
            truncation,
            padding,
        })
    }
}
