use anyhow::anyhow;

use crate::{
    decoder::Decoder,
    model::WordPiece,
    normalizer::Normalizer,
    padding::Padding,
    post_tokenizer::PostTokenizer,
    pre_tokenizer::PreTokenizer,
    tokenizer::Tokenizer,
    truncation::Truncation,
    Error,
};

pub struct Builder {
    normalizer: Normalizer,
    pre_tokenizer: Option<PreTokenizer>,
    model: Option<WordPiece>,
    post_tokenizer: Option<PostTokenizer>,
    decoder: Option<Decoder>,
    truncation: Truncation,
    padding: Padding,
}

impl Builder {
    /// Get an empty Builder.
    pub fn new() -> Self {
        Self {
            normalizer: Normalizer::None,
            pre_tokenizer: None,
            model: None,
            post_tokenizer: None,
            decoder: None,
            truncation: Truncation::None,
            padding: Padding::None,
        }
    }

    /// Set the normalizer.
    pub fn with_normalizer(mut self, normalizer: Normalizer) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Set the pre-tokenizer.
    pub fn with_pre_tokenizer(mut self, pretokenizer: Option<PreTokenizer>) -> Self {
        self.pre_tokenizer = pretokenizer;
        self
    }

    /// Set the model.
    pub fn with_model(mut self, model: WordPiece) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the post-processor.
    pub fn with_post_tokenizer(mut self, post_tokenizer: Option<PostTokenizer>) -> Self {
        self.post_tokenizer = post_tokenizer;
        self
    }

    /// Set the decoder.
    pub fn with_decoder(mut self, decoder: Option<Decoder>) -> Self {
        self.decoder = decoder;
        self
    }

    /// Set the trunaction parameters.
    pub fn with_truncation(mut self, trunc: Truncation) -> Self {
        self.truncation = trunc;
        self
    }

    /// Set the padding parameters.
    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    /// Convert the Builder to a Tokenizer.
    ///
    /// Conversion fails if the `model` is missing.
    pub fn build(self) -> Result<Tokenizer, Error> {
        Ok(Tokenizer {
            normalizer: self.normalizer,
            pre_tokenizer: self.pre_tokenizer,
            model: self.model.ok_or_else(|| anyhow!("Model missing."))?,
            post_tokenizer: self.post_tokenizer,
            decoder: self.decoder,
            truncation: self.truncation,
            padding: self.padding,
        })
    }
}
