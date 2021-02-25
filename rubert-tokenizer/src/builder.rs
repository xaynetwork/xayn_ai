use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::{
    model::{Model, Vocab},
    normalizer::Normalizer,
    padding::Padding,
    post_tokenizer::PostTokenizer,
    pre_tokenizer::PreTokenizer,
    tokenizer::Tokenizer,
    truncation::Truncation,
    Error,
};

/// A builder to create a [`Tokenizer`].
pub struct Builder {
    vocab: Vocab,
    unk: String,
    prefix: String,
    max_chars: usize,
    normalizer: Normalizer,
    pre_tokenizer: PreTokenizer,
    post_tokenizer: PostTokenizer,
    truncation: Truncation,
    padding: Padding,
}

impl Builder {
    /// Creates a [`Tokenizer`] builder from an in-memory vocabulary.
    ///
    /// The default settings are:
    /// - A Bert word piece model with `"[UNK]"` unknown token, `"##"` continuing subword prefix
    /// and `100` maximum characters per word.
    /// - The default [`Normalizer`], [`PreTokenizer`] and [`PostTokenizer`].
    /// - The default [`Truncation`] and [`Padding`] stategies.
    ///
    /// # Errors
    /// Fails on invalid vocabularies.
    pub fn new(vocab: impl BufRead) -> Result<Self, Error> {
        Ok(Self {
            vocab: Model::parse_vocab(vocab)?,
            unk: "[UNK]".into(),
            prefix: "##".into(),
            max_chars: 100,
            normalizer: Normalizer::default(),
            pre_tokenizer: PreTokenizer::default(),
            post_tokenizer: PostTokenizer::default(),
            truncation: Truncation::default(),
            padding: Padding::default(),
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
    pub fn with_normalizer(mut self, normalizer: Normalizer) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Configures the pre-tokenizer.
    pub fn with_pre_tokenizer(mut self, pre_tokenizer: PreTokenizer) -> Self {
        self.pre_tokenizer = pre_tokenizer;
        self
    }

    /// Configures the model.
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
    /// # Errors
    /// Fails on invalid post-tokenizer configurations.
    pub fn with_post_tokenizer(mut self, post_tokenizer: PostTokenizer) -> Result<Self, Error> {
        self.post_tokenizer = post_tokenizer.validate(&self.vocab)?;
        Ok(self)
    }

    /// Configures the truncation strategy.
    ///
    /// # Errors
    /// Fails on invalid truncation configurations.
    pub fn with_truncation(mut self, truncation: Truncation) -> Result<Self, Error> {
        self.truncation = truncation.validate()?;
        Ok(self)
    }

    /// Configures the padding strategy.
    ///
    /// # Errors
    /// Fails on invalid padding configurations.
    pub fn with_padding(mut self, padding: Padding) -> Result<Self, Error> {
        self.padding = padding.validate(&self.vocab)?;
        Ok(self)
    }

    /// Builds the tokenizer.
    ///
    /// # Errors
    /// Fails on invalid model configurations.
    pub fn build(self) -> Result<Tokenizer, Error> {
        let model = Model {
            vocab: self.vocab,
            unk_id: 0,
            unk_token: self.unk,
            continuing_subword_prefix: self.prefix,
            max_input_chars_per_word: self.max_chars,
        }
        .validate()?;
        Ok(Tokenizer {
            normalizer: self.normalizer,
            pre_tokenizer: self.pre_tokenizer,
            model,
            post_tokenizer: self.post_tokenizer,
            truncation: self.truncation,
            padding: self.padding,
        })
    }
}
