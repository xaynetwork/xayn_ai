use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::{
    model::{Model, Vocab},
    normalizer::Normalizer,
    post_tokenizer::{padding::Padding, truncation::Truncation, PostTokenizer},
    pre_tokenizer::PreTokenizer,
    tokenizer::Tokenizer,
    Error,
};

/// A builder to create a [`Tokenizer`].
pub struct Builder {
    // normalizer
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: bool,
    lowercase: bool,
    // model
    vocab: Vocab,
    unk: String,
    prefix: String,
    max_chars: usize,
    // post-tokenizer
    cls: String,
    sep: String,
    trunc: Truncation,
    pad: Padding,
}

impl Builder {
    /// Creates a [`Tokenizer`] builder from an in-memory vocabulary.
    ///
    /// # Errors
    /// Fails on invalid vocabularies.
    pub fn new(vocab: impl BufRead) -> Result<Self, Error> {
        Ok(Self {
            // normalizer
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: true,
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
    /// The default settings are the same as for [`new()`].
    ///
    /// # Errors
    /// Fails on invalid vocabularies.
    pub fn from_file(vocab: impl AsRef<Path>) -> Result<Self, Error> {
        Self::new(BufReader::new(File::open(vocab)?))
    }

    /// Configures the normalizer.
    ///
    /// Configurable by:
    /// - Cleans any control characters and replaces all sorts of whitespace by ` `. Defaults to
    /// `true`.
    /// - Puts spaces around chinese characters so they get split. Defaults to `true`.
    /// - Strips accents from characters. Defaults to `true`.
    /// - Lowercases characters. Defaults to `true`.
    pub fn with_normalizer(
        mut self,
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: bool,
        lowercase: bool,
    ) -> Self {
        self.clean_text = clean_text;
        self.handle_chinese_chars = handle_chinese_chars;
        self.strip_accents = strip_accents;
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
        unk: impl Into<String>,
        prefix: impl Into<String>,
        max_chars: usize,
    ) -> Self {
        self.unk = unk.into();
        self.prefix = prefix.into();
        self.max_chars = max_chars;
        self
    }

    /// Configures the post-tokenizer.
    ///
    /// Configurable by:
    /// - The class token. Defaults to `"[CLS]"`.
    /// - The separation token. Defaults to `"[SEP]"`.
    pub fn with_post_tokenizer(mut self, cls: impl Into<String>, sep: impl Into<String>) -> Self {
        self.cls = cls.into();
        self.sep = sep.into();
        self
    }

    /// Configures the truncation strategy.
    ///
    /// Defaults to no truncation.
    ///
    /// # Errors
    /// Fails on invalid truncation configurations.
    pub fn with_truncation(mut self, trunc: Truncation) -> Self {
        self.trunc = trunc;
        self
    }

    /// Configures the padding strategy.
    ///
    /// Defaults to no padding.
    ///
    /// # Errors
    /// Fails on invalid padding configurations.
    pub fn with_padding(mut self, pad: Padding) -> Self {
        self.pad = pad;
        self
    }

    /// Builds the tokenizer.
    ///
    /// # Errors
    /// Fails on invalid model configurations.
    pub fn build(self) -> Result<Tokenizer, Error> {
        let normalizer = Normalizer::new(
            self.clean_text,
            self.handle_chinese_chars,
            self.strip_accents,
            self.lowercase,
        );
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
