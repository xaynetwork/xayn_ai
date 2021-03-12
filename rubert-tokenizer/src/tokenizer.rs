use num_traits::{FromPrimitive, Num};

use crate::{
    model::Model,
    normalizer::Normalizer,
    post_tokenizer::{encoding::Encoding, padding::Padding, truncation::Truncation, PostTokenizer},
    pre_tokenizer::PreTokenizer,
};

/// A Bert tokenizer.
///
/// Can be created via the [`Builder`] and consists of a Bert normalizer, a Bert pre-tokenizer, a
/// Bert word piece model and a Bert post-tokenizer including truncation and padding strategies.
///
/// [`Builder`]: crate::Builder
pub struct Tokenizer<N> {
    pub(crate) normalizer: Normalizer,
    pub(crate) pre_tokenizer: PreTokenizer,
    pub(crate) model: Model<N>,
    pub(crate) post_tokenizer: PostTokenizer<N>,
    pub(crate) truncation: Truncation,
    pub(crate) padding: Padding<N>,
}

impl<N> Tokenizer<N> {
    /// Encodes the sequence.
    pub fn encode(&self, sequence: impl AsRef<str>) -> Encoding<N>
    where
        N: Num + FromPrimitive + Copy,
    {
        let sequence = self.normalizer.normalize(sequence);
        let sequence = self.pre_tokenizer.pre_tokenize(sequence);
        let sequence = self.model.tokenize(sequence);

        let encoding = self.truncation.truncate(sequence.into());
        let encoding = self.post_tokenizer.post_tokenize(encoding);
        self.padding.pad(encoding)
    }

    /// Encodes the sequences.
    pub fn encode_batch(&self, sequences: &[impl AsRef<str>]) -> Vec<Encoding<N>>
    where
        N: Num + FromPrimitive + Copy,
    {
        sequences
            .iter()
            .map(|sequence| self.encode(sequence))
            .collect()
    }

    /// Decodes the encoding with optional cleanup.
    pub fn decode(&self, encoding: &Encoding<N>, cleanup: bool) -> String {
        encoding.decode(
            self.model.unk_token.as_str(),
            self.model.prefix.as_str(),
            cleanup,
        )
    }

    /// Decodes the encodings with optional cleanup.
    pub fn decode_batch(&self, encodings: &[Encoding<N>], cleanup: bool) -> Vec<String> {
        encodings
            .iter()
            .map(|encoding| self.decode(encoding, cleanup))
            .collect()
    }
}
