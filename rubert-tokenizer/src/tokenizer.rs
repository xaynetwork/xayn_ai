use std::iter::IntoIterator;

use crate::{
    model::{string::TokenizedString, Model},
    normalizer::{string::NormalizedString, Normalizer},
    post_tokenizer::{encoding::Encoding, padding::Padding, truncation::Truncation, PostTokenizer},
    pre_tokenizer::{string::PreTokenizedString, PreTokenizer},
    Error,
};

/// A Bert tokenizer.
///
/// Can be created via the [`Builder`] and consists of a Bert normalizer, a Bert pre-tokenizer, a
/// Bert word piece model and a Bert post-tokenizer including truncation and padding strategies.
///
/// [`Builder`]: crate::Builder
pub struct Tokenizer {
    pub(crate) normalizer: Normalizer,
    pub(crate) pre_tokenizer: PreTokenizer,
    pub(crate) model: Model,
    pub(crate) post_tokenizer: PostTokenizer,
    pub(crate) truncation: Truncation,
    pub(crate) padding: Padding,
}

impl Tokenizer {
    /// Normalizes the sequence.
    fn normalize(&self, sequence: impl AsRef<str>) -> NormalizedString {
        self.normalizer.normalize(sequence)
    }

    /// Pre-tokenizes the sequence
    fn pre_tokenize(&self, sequence: NormalizedString) -> Result<PreTokenizedString, Error> {
        self.pre_tokenizer.pre_tokenize(sequence)
    }

    /// Tokenizes the sequence.
    fn tokenize(&self, sequence: PreTokenizedString) -> Result<TokenizedString, Error> {
        self.model.tokenize(sequence)
    }

    /// Post-tokenizes the sequence
    fn post_tokenize(&self, sequence: TokenizedString) -> Encoding {
        let encoding = self
            .truncation
            .truncate(sequence.into(), PostTokenizer::ADDED_TOKENS);
        let encoding = self.post_tokenizer.post_tokenize(encoding);
        self.padding.pad(encoding)
    }

    /// Encodes the sequence.
    pub fn encode(&self, sequence: impl AsRef<str>) -> Result<Encoding, Error> {
        let normalized = self.normalize(sequence);
        let pre_tokenized = self.pre_tokenize(normalized)?;
        let tokenized = self.tokenize(pre_tokenized)?;
        // TODO: move into encoding to post-tokenizer
        let encoding = self.post_tokenize(tokenized);

        Ok(encoding)
    }

    /// Encodes the sequences.
    pub fn encode_batch(&self, sequences: &[impl AsRef<str>]) -> Result<Vec<Encoding>, Error> {
        sequences
            .into_iter()
            .map(|sequence| self.encode(sequence))
            .collect()
    }

    /// Decodes the encoding with optional cleanup.
    pub fn decode(&self, encoding: &Encoding, cleanup: bool) -> String {
        encoding.decode(
            self.model.unk_token.as_str(),
            self.model.prefix.as_str(),
            cleanup,
        )
    }

    /// Decodes the encodings with optional cleanup.
    pub fn decode_batch(&self, encodings: &[Encoding], cleanup: bool) -> Vec<String> {
        encodings
            .iter()
            .map(|encoding| self.decode(encoding, cleanup))
            .collect()
    }
}
