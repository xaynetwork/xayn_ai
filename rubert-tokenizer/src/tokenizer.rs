use std::iter::IntoIterator;

use crate::{
    model::{encoding::Encoding, string::TokenizedString, Model},
    normalizer::{string::NormalizedString, Normalizer},
    post_tokenizer::{padding::Padding, truncation::Truncation, PostTokenizer},
    pre_tokenizer::{string::PreTokenizedString, PreTokenizer},
    Error,
};

/// A Bert tokenizer.
///
/// Can be created via the [`Builder`] and consists of a normalizer, a pre-tokenizer, a Bert word
/// piece model and a post-tokenizer including truncation and padding strategies.
///
/// [`Builer`]: crate::Builder
pub struct Tokenizer {
    pub(crate) normalizer: Normalizer,
    pub(crate) pre_tokenizer: PreTokenizer,
    pub(crate) model: Model,
    pub(crate) post_tokenizer: PostTokenizer,
    pub(crate) truncation: Truncation,
    pub(crate) padding: Padding,
}

impl Tokenizer {
    /// Normalization logic, go through all normalizers
    fn normalize(&self, sequence: impl AsRef<str>) -> NormalizedString {
        self.normalizer.normalize(sequence)
    }

    /// PreTokenization logic, handling the case where there is no PreTokenizer set
    fn pre_tokenize(&self, sequence: NormalizedString) -> Result<PreTokenizedString, Error> {
        self.pre_tokenizer.pre_tokenize(sequence)
    }

    /// Tokenization logic, makes the bridge between the pre-tokenization phase and the real
    /// tokenization phase, and converting offsets back to the original referential.
    fn tokenize(&self, sequence: PreTokenizedString) -> Result<TokenizedString, Error> {
        self.model.tokenize(sequence)
    }

    /// Post processing logic, handling the case where there is no PostProcessor set
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

    /// Decodes an encoding.
    pub fn decode(&self, encoding: &Encoding, cleanup: bool) -> String {
        encoding.decode(
            self.model.unk_token.as_str(),
            self.model.prefix.as_str(),
            cleanup,
        )
    }

    /// Decodes the encodings.
    pub fn decode_batch(&self, encodings: &[Encoding], cleanup: bool) -> Vec<String> {
        encodings
            .iter()
            .map(|encoding| self.decode(encoding, cleanup))
            .collect()
    }
}
