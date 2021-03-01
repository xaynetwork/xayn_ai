use std::iter::IntoIterator;

use crate::{
    encoding::Encoding,
    model::Model,
    normalizer::{NormalizedString, Normalizer, Offsets},
    padding::Padding,
    post_tokenizer::PostTokenizer,
    pre_tokenizer::{OffsetType, PreTokenizedString, PreTokenizer},
    truncation::Truncation,
    Error,
};

pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: Offsets,
}

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
    fn pre_tokenize(
        &self,
        normalized: impl Into<PreTokenizedString>,
    ) -> Result<PreTokenizedString, Error> {
        self.pre_tokenizer.pre_tokenize(normalized)
    }

    /// Tokenization logic, makes the bridge between the pre-tokenization phase and the real
    /// tokenization phase, and converting offsets back to the original referential.
    fn tokenize(
        &self,
        pre_tokenized: impl Into<PreTokenizedString>,
        type_id: u32,
        word_idx: Option<u32>,
        offsets_type: OffsetType,
    ) -> Result<Encoding, Error> {
        pre_tokenized
            .into()
            .tokenize(|normalized| self.model.tokenize(normalized.normalized.as_str()))?
            .into_encoding(word_idx, type_id, offsets_type)
    }

    /// Post processing logic, handling the case where there is no PostProcessor set
    fn post_process(&self, encoding: Encoding) -> Encoding {
        let encoding = self
            .truncation
            .truncate(encoding, self.post_tokenizer.added_tokens());
        let encoding = self.post_tokenizer.process(encoding);
        self.padding.pad(encoding)
    }

    /// Encode a single sequence
    fn encode_single_sequence(
        &self,
        sequence: impl AsRef<str>,
        type_id: u32,
        offsets_type: OffsetType,
    ) -> Result<Encoding, Error> {
        let normalized = self.normalize(sequence);
        let pre_tokenized = self.pre_tokenize(normalized)?;
        let tokenized = self.tokenize(pre_tokenized, type_id, None, offsets_type);
        tokenized.map(|tokenized| self.post_process(tokenized))
    }

    /// Encode the given input. This method accepts both single sequences, as well as pair
    /// sequences. Also, a sequence can be a string, or already pre-tokenized input directly:
    ///
    /// ```
    /// # use tokenizers::Tokenizer;
    /// # use tokenizers::models::bpe::BPE;
    /// # let mut tokenizer = Tokenizer::new(BPE::default());
    /// #
    /// // Sequences:
    /// tokenizer.encode("Single sequence", false);
    /// tokenizer.encode(("Sequence A", "Sequence B"), false);
    ///
    /// // Pre-tokenized sequences:
    /// tokenizer.encode(&["Single", "sequence"][..], false);
    /// tokenizer.encode((&["Sequence", "A"][..], &["Sequence", "B"][..]), false);
    ///
    /// // or even both types together:
    /// tokenizer.encode(
    ///     ("A complete sequence", &["And", "a", "tokenized"][..]),
    ///     false,
    /// );
    /// ```
    pub fn encode(&self, sequence: impl AsRef<str>) -> Result<Encoding, Error> {
        self.encode_single_sequence(sequence, 0, OffsetType::Byte)
    }

    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch(&self, sequences: &[impl AsRef<str>]) -> Result<Vec<Encoding>, Error> {
        sequences
            .into_iter()
            .map(|sequence| self.encode(sequence))
            .collect()
    }

    /// Encode the given input, using offsets relative to chars instead of bytes.
    /// This method accepts both single sequences, as well as pair sequences. Also,
    /// a sequence can be a string, or already pre-tokenized input directly:
    ///
    /// ```
    /// # use tokenizers::Tokenizer;
    /// # use tokenizers::models::bpe::BPE;
    /// # let mut tokenizer = Tokenizer::new(BPE::default());
    /// #
    /// // Sequences:
    /// tokenizer.encode("Single sequence", false);
    /// tokenizer.encode(("Sequence A", "Sequence B"), false);
    ///
    /// // Pre-tokenized sequences:
    /// tokenizer.encode(&["Single", "sequence"][..], false);
    /// tokenizer.encode((&["Sequence", "A"][..], &["Sequence", "B"][..]), false);
    ///
    /// // or even both types together:
    /// tokenizer.encode(
    ///     ("A complete sequence", &["And", "a", "tokenized"][..]),
    ///     false,
    /// );
    /// ```
    pub fn encode_char_offsets(&self, sequence: impl AsRef<str>) -> Result<Encoding, Error> {
        self.encode_single_sequence(sequence, 0, OffsetType::Char)
    }

    /// Encode all the sentences in parallel, using multiple threads.
    /// The offsets on each `Encoding` will be relative to chars instead of bytes.
    pub fn encode_batch_char_offsets(
        &self,
        sequences: &[impl AsRef<str>],
    ) -> Result<Vec<Encoding>, Error> {
        sequences
            .into_iter()
            .map(|sequence| self.encode_char_offsets(sequence))
            .collect()
    }

    /// Decodes an encoding back to a String.
    pub fn decode(&self, encoding: &Encoding, cleanup: bool) -> String {
        self.model.de_tokenize(encoding, cleanup)
    }

    /// Decodes the encodings back to strings.
    pub fn decode_batch(&self, encodings: &[Encoding], cleanup: bool) -> Vec<String> {
        encodings
            .iter()
            .map(|encoding| self.decode(encoding, cleanup))
            .collect()
    }
}
