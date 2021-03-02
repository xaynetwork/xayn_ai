use std::{convert::TryInto, iter::IntoIterator};

use crate::{
    model::{encoding::Encoding, Model},
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
    fn pre_tokenize(&self, normalized: NormalizedString) -> Result<PreTokenizedString, Error> {
        self.pre_tokenizer.pre_tokenize(normalized)
    }

    /// Tokenization logic, makes the bridge between the pre-tokenization phase and the real
    /// tokenization phase, and converting offsets back to the original referential.
    fn tokenize(&self, pre_tokenized: PreTokenizedString) -> Result<Encoding, Error> {
        self.model.tokenize(pre_tokenized)?.try_into()
    }

    /// Post processing logic, handling the case where there is no PostProcessor set
    fn post_tokenize(&self, encoding: Encoding) -> Encoding {
        let encoding = self
            .truncation
            .truncate(encoding, PostTokenizer::ADDED_TOKENS);
        let encoding = self.post_tokenizer.process(encoding);
        self.padding.pad(encoding)
    }

    /// Encodes the sequence.
    pub fn encode(&self, sequence: impl AsRef<str>) -> Result<Encoding, Error> {
        let string = self.normalize(sequence);
        let string = self.pre_tokenize(string)?;
        let encoding = self.tokenize(string)?;
        let encoding = self.post_tokenize(encoding);

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
