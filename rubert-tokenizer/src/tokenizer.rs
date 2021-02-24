use std::iter::IntoIterator;

use crate::{
    decoder::WordPieceDecoder,
    encoding::Encoding,
    model::{Vocab, WordPiece},
    normalizer::{NormalizedString, Normalizer, Offsets},
    padding::Padding,
    post_tokenizer::PostTokenizer,
    pre_tokenizer::{OffsetType, PreTokenizedString, PreTokenizer},
    sequence::Sequence,
    truncation::Truncation,
    Error,
};

pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: Offsets,
}

pub struct Tokenizer {
    // Tokenizer parts
    pub(crate) normalizer: Option<Normalizer>,
    pub(crate) pre_tokenizer: Option<PreTokenizer>,
    pub(crate) model: WordPiece,
    pub(crate) post_tokenizer: Option<PostTokenizer>,
    pub(crate) decoder: Option<WordPieceDecoder>,
    // General processing parameters
    pub(crate) truncation: Truncation,
    pub(crate) padding: Padding,
}

impl Tokenizer {
    /// Get the vocabulary
    pub fn vocab(&self) -> &Vocab {
        self.model.vocab()
    }

    /// Get the size of the vocabulary
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.token_to_id(token)
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.id_to_token(id)
    }

    /// Normalization logic, go through all normalizers
    fn normalize(&self, sequence: impl Into<NormalizedString>) -> Result<NormalizedString, Error> {
        if let Some(ref normalizer) = self.normalizer {
            normalizer.normalize(sequence)
        } else {
            Ok(sequence.into())
        }
    }

    /// PreTokenization logic, handling the case where there is no PreTokenizer set
    fn pre_tokenize(
        &self,
        normalized: impl Into<PreTokenizedString>,
    ) -> Result<PreTokenizedString, Error> {
        if let Some(ref pretokenizer) = self.pre_tokenizer {
            pretokenizer.pre_tokenize(normalized)
        } else {
            Ok(normalized.into())
        }
    }

    /// Tokenization logic, makes the bridge between the pre-tokenization phase and the real
    /// tokenization phase, and converting offsets back to the original referential.
    fn tokenize(
        &self,
        pretokenized: impl Into<PreTokenizedString>,
        type_id: u32,
        word_idx: Option<u32>,
        offsets_type: OffsetType,
    ) -> Result<Encoding, Error> {
        pretokenized
            .into()
            .tokenize(|normalized| self.model.tokenize(normalized.normalized.as_str()))?
            .into_encoding(word_idx, type_id, offsets_type)
    }

    /// Post processing logic, handling the case where there is no PostProcessor set
    fn post_process(
        &self,
        encoding: Encoding,
        add_special_tokens: bool,
    ) -> Result<Encoding, Error> {
        // 1. First we truncate if needed
        let added_tokens = self
            .post_tokenizer
            .as_ref()
            .map(|_| PostTokenizer::ADDED_TOKENS)
            .unwrap_or_default();
        let encoding = self.truncation.truncate_encoding(encoding, added_tokens);

        // 2. Then we post-process
        let final_encoding = if let Some(ref processor) = self.post_tokenizer {
            processor.process(encoding, add_special_tokens)
        } else {
            encoding
        };

        // 3. Then we pad if needed
        let final_encoding = self.padding.pad_encoding(final_encoding);

        Ok(final_encoding)
    }

    /// Encode a single sequence
    fn encode_single_sequence<'s>(
        &self,
        sequence: impl Into<Sequence<'s>>,
        type_id: u32,
        offsets_type: OffsetType,
    ) -> Result<Encoding, Error> {
        let encode = |is_pre_tokenized: bool,
                      sequence_idx: usize,
                      sequence: &str|
         -> Result<Encoding, Error> {
            let normalized = self.normalize(sequence)?;
            let pre_tokenized = self.pre_tokenize(normalized)?;
            self.tokenize(
                pre_tokenized,
                type_id,
                if is_pre_tokenized {
                    Some(sequence_idx as u32)
                } else {
                    None
                },
                offsets_type,
            )
        };

        match sequence.into() {
            Sequence::Raw(sequence) => encode(false, 0, sequence.as_ref()),
            Sequence::PreTokenized(sequence) => sequence
                .into_iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            Sequence::PreTokenizedOwned(sequence) => sequence
                .into_iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            Sequence::PreTokenizedCow(sequence) => sequence
                .into_iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
        }
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
    pub fn encode<'s>(
        &self,
        sequence: impl Into<Sequence<'s>>,
        add_special_tokens: bool,
    ) -> Result<Encoding, Error> {
        let encoding = self.encode_single_sequence(sequence, 0, OffsetType::Byte)?;
        self.post_process(encoding, add_special_tokens)
    }

    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch<'s>(
        &self,
        sequences: Vec<impl Into<Sequence<'s>>>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, Error> {
        let encodings = sequences
            .into_iter()
            .map(|sequence| self.encode(sequence, add_special_tokens))
            .collect::<Result<Vec<Encoding>, Error>>()?;

        // We do the padding here to make sure we handle the batch padding
        let encodings = self.padding.pad_encodings(encodings);

        Ok(encodings)
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
    pub fn encode_char_offsets<'s>(
        &self,
        sequence: impl Into<Sequence<'s>>,
        add_special_tokens: bool,
    ) -> Result<Encoding, Error> {
        let encoding = self.encode_single_sequence(sequence, 0, OffsetType::Char)?;
        self.post_process(encoding, add_special_tokens)
    }

    /// Encode all the sentences in parallel, using multiple threads.
    /// The offsets on each `Encoding` will be relative to chars instead of bytes.
    pub fn encode_batch_char_offsets<'s>(
        &self,
        sequences: Vec<impl Into<Sequence<'s>>>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, Error> {
        let encodings = sequences
            .into_iter()
            .map(|sequence| self.encode_char_offsets(sequence, add_special_tokens))
            .collect::<Result<Vec<Encoding>, Error>>()?;

        // We do the padding here to make sure we handle the batch padding
        let encodings = self.padding.pad_encodings(encodings);

        Ok(encodings)
    }

    /// Decode the given ids, back to a String
    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String, Error> {
        let tokens = ids
            .into_iter()
            .filter_map(|id| {
                self.model
                    .id_to_token(id)
                    .filter(|token| !skip_special_tokens || token != self.model.unk_token.as_str())
            })
            .collect::<Vec<_>>();

        if let Some(decoder) = &self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(" "))
        }
    }

    /// Decode all sentences in parallel
    pub fn decode_batch(
        &self,
        sentences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, Error> {
        sentences
            .into_iter()
            .map(|sentence| self.decode(sentence, skip_special_tokens))
            .collect()
    }
}
