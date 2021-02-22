use std::{borrow::Cow, collections::HashMap, iter::IntoIterator};

use anyhow::anyhow;

use crate::{
    added_vocabulary::{AddedToken, AddedVocabulary},
    decoder::WordPieceDecoder,
    encoding::Encoding,
    model::WordPiece,
    normalizer::{normalized_string::NormalizedString, BertNormalizer},
    padding::Padding,
    pre_tokenizer::{BertPreTokenizer, OffsetType, PreTokenizedString},
    processor::BertProcessing,
    truncation::{truncate_encodings, TruncationParams},
    Error,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
}
impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize)) -> Self {
        Token { id, value, offsets }
    }
}

pub enum InputSequence<'s> {
    Raw(Cow<'s, str>),
    PreTokenized(Cow<'s, [&'s str]>),
    PreTokenizedOwned(Cow<'s, [String]>),
    PreTokenizedCow(Cow<'s, [Cow<'s, str>]>),
}

impl<'s> From<Cow<'s, str>> for InputSequence<'s> {
    fn from(input: Cow<'s, str>) -> Self {
        InputSequence::Raw(input)
    }
}

impl<'s> From<&'s str> for InputSequence<'s> {
    fn from(input: &'s str) -> Self {
        InputSequence::Raw(Cow::Borrowed(input))
    }
}

impl From<String> for InputSequence<'_> {
    fn from(input: String) -> Self {
        InputSequence::Raw(Cow::Owned(input))
    }
}

impl<'s> From<&'s [&'s str]> for InputSequence<'s> {
    fn from(input: &'s [&'s str]) -> Self {
        InputSequence::PreTokenized(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<&'s str>> for InputSequence<'s> {
    fn from(input: Vec<&'s str>) -> Self {
        InputSequence::PreTokenized(Cow::Owned(input))
    }
}

impl<'s> From<&'s [String]> for InputSequence<'s> {
    fn from(input: &'s [String]) -> Self {
        InputSequence::PreTokenizedOwned(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<String>> for InputSequence<'s> {
    fn from(input: Vec<String>) -> Self {
        InputSequence::PreTokenizedOwned(Cow::Owned(input))
    }
}

impl<'s> From<Vec<Cow<'s, str>>> for InputSequence<'s> {
    fn from(input: Vec<Cow<'s, str>>) -> Self {
        InputSequence::PreTokenizedCow(Cow::Owned(input))
    }
}

impl<'s> From<&'s [Cow<'s, str>]> for InputSequence<'s> {
    fn from(input: &'s [Cow<'s, str>]) -> Self {
        InputSequence::PreTokenizedCow(Cow::Borrowed(input))
    }
}

pub enum EncodeInput<'s> {
    Single(InputSequence<'s>),
    Dual(InputSequence<'s>, InputSequence<'s>),
}

impl<'s, I: Into<InputSequence<'s>>> From<I> for EncodeInput<'s> {
    fn from(input: I) -> Self {
        EncodeInput::Single(input.into())
    }
}

impl<'s, I1, I2> From<(I1, I2)> for EncodeInput<'s>
where
    I1: Into<InputSequence<'s>>,
    I2: Into<InputSequence<'s>>,
{
    fn from(input: (I1, I2)) -> Self {
        EncodeInput::Dual(input.0.into(), input.1.into())
    }
}

pub struct TokenizerBuilder {
    model: Option<WordPiece>,
    normalizer: Option<BertNormalizer>,
    pre_tokenizer: Option<BertPreTokenizer>,
    post_processor: Option<BertProcessing>,
    decoder: Option<WordPieceDecoder>,

    added_vocabulary: AddedVocabulary,

    truncation: Option<TruncationParams>,
    padding: Option<Padding>,
}

impl TokenizerBuilder {
    /// Get an empty TokenizerBuilder.
    pub fn new() -> Self {
        TokenizerBuilder {
            model: None,
            normalizer: None,
            pre_tokenizer: None,
            post_processor: None,
            decoder: None,
            added_vocabulary: AddedVocabulary::new(),
            truncation: None,
            padding: None,
        }
    }

    /// Convert the TokenizerBuilder to a Tokenizer.
    ///
    /// Conversion fails if the `model` is missing.
    pub fn build(self) -> Result<TokenizerImpl, Error> {
        let model = self.model.ok_or_else(|| anyhow!("Model missing."))?;
        Ok(TokenizerImpl {
            normalizer: self.normalizer,
            pre_tokenizer: self.pre_tokenizer,
            model,

            post_processor: self.post_processor,
            decoder: self.decoder,
            added_vocabulary: self.added_vocabulary,
            truncation: self.truncation,
            padding: self.padding,
        })
    }

    /// Set the model.
    pub fn with_model(mut self, model: WordPiece) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the normalizer.
    pub fn with_normalizer(mut self, normalizer: Option<BertNormalizer>) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Set the pre-tokenizer.
    pub fn with_pre_tokenizer(mut self, pretokenizer: Option<BertPreTokenizer>) -> Self {
        self.pre_tokenizer = pretokenizer;
        self
    }

    /// Set the post-processor.
    pub fn with_post_processor(mut self, post_processor: Option<BertProcessing>) -> Self {
        self.post_processor = post_processor;
        self
    }

    /// Set the decoder.
    pub fn with_decoder(mut self, decoder: Option<WordPieceDecoder>) -> Self {
        self.decoder = decoder;
        self
    }

    /// Set the trunaction parameters.
    pub fn with_truncation(mut self, trunc: Option<TruncationParams>) -> Self {
        self.truncation = trunc;
        self
    }

    /// Set the padding parameters.
    pub fn with_padding(mut self, padding: Option<Padding>) -> Self {
        self.padding = padding;
        self
    }
}

pub struct TokenizerImpl {
    // Tokenizer parts
    normalizer: Option<BertNormalizer>,
    pre_tokenizer: Option<BertPreTokenizer>,
    model: WordPiece,
    post_processor: Option<BertProcessing>,
    decoder: Option<WordPieceDecoder>,

    // Added Vocabulary capabilities
    added_vocabulary: AddedVocabulary,

    // General processing parameters
    truncation: Option<TruncationParams>,
    padding: Option<Padding>,
}

impl TokenizerImpl {
    /// Instantiate a new Tokenizer, with the given Model
    pub fn new(model: WordPiece) -> Self {
        TokenizerImpl {
            normalizer: None,
            pre_tokenizer: None,
            model,
            post_processor: None,
            decoder: None,

            added_vocabulary: AddedVocabulary::new(),

            truncation: None,
            padding: None,
        }
    }

    /// Set the normalizer
    pub fn with_normalizer(&mut self, normalizer: impl Into<BertNormalizer>) -> &mut Self {
        self.normalizer = Some(normalizer.into());
        self
    }

    /// Get the normalizer
    pub fn get_normalizer(&self) -> Option<&BertNormalizer> {
        self.normalizer.as_ref()
    }

    /// Set the pre tokenizer
    pub fn with_pre_tokenizer(&mut self, pre_tokenizer: impl Into<BertPreTokenizer>) -> &mut Self {
        self.pre_tokenizer = Some(pre_tokenizer.into());
        self
    }

    /// Get the pre tokenizer
    pub fn get_pre_tokenizer(&self) -> Option<&BertPreTokenizer> {
        self.pre_tokenizer.as_ref()
    }

    /// Set the post processor
    pub fn with_post_processor(&mut self, post_processor: impl Into<BertProcessing>) -> &mut Self {
        self.post_processor = Some(post_processor.into());
        self
    }

    /// Get the post processor
    pub fn get_post_processor(&self) -> Option<&BertProcessing> {
        self.post_processor.as_ref()
    }

    /// Set the decoder
    pub fn with_decoder(&mut self, decoder: impl Into<WordPieceDecoder>) -> &mut Self {
        self.decoder = Some(decoder.into());
        self
    }

    /// Get the decoder
    pub fn get_decoder(&self) -> Option<&WordPieceDecoder> {
        self.decoder.as_ref()
    }

    /// Set the model
    pub fn with_model(&mut self, model: impl Into<WordPiece>) -> &mut Self {
        self.model = model.into();
        self
    }

    /// Get the model
    pub fn get_model(&self) -> &WordPiece {
        &self.model
    }

    /// Set the truncation parameters
    pub fn with_truncation(&mut self, trunc: Option<TruncationParams>) -> &mut Self {
        self.truncation = trunc;
        self
    }

    /// Get the currently set truncation parameters
    pub fn get_truncation(&self) -> Option<&TruncationParams> {
        self.truncation.as_ref()
    }

    /// Get a mutable reference to the currently set truncation parameters
    pub fn get_truncation_mut(&mut self) -> Option<&mut TruncationParams> {
        self.truncation.as_mut()
    }

    /// Set the padding parameters
    pub fn with_padding(&mut self, padding: Option<Padding>) -> &mut Self {
        self.padding = padding;
        self
    }

    /// Get the currently set padding parameters
    pub fn get_padding(&self) -> Option<&Padding> {
        self.padding.as_ref()
    }

    /// Get a mutable reference to the currently set padding parameters
    pub fn get_padding_mut(&mut self) -> Option<&mut Padding> {
        self.padding.as_mut()
    }

    /// Get the vocabulary
    pub fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        let mut final_vocab = self.model.get_vocab();

        if with_added_tokens {
            let added_vocab = self.added_vocabulary.get_vocab();
            if !added_vocab.is_empty() {
                final_vocab.reserve(added_vocab.len());
                for (token, id) in added_vocab {
                    final_vocab.insert(token.clone(), *id);
                }
            }
        }

        final_vocab
    }

    /// Get the size of the vocabulary
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.model.get_vocab_size()
            + if with_added_tokens {
                self.added_vocabulary.len()
            } else {
                0
            }
    }

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added_vocabulary.token_to_id(token, &self.model)
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.added_vocabulary.id_to_token(id, &self.model)
    }

    /// Encode a single sequence
    fn encode_single_sequence(
        &self,
        sequence: InputSequence,
        type_id: u32,
        offsets_type: OffsetType,
    ) -> Result<Encoding, Error> {
        let encode = |is_pre_tokenized, subseq_idx, subseq| -> Result<Encoding, Error> {
            let normalized = self
                .added_vocabulary
                .extract_and_normalize(self.normalizer.as_ref(), subseq);
            let pre_tokenized = self.do_pre_tokenize(normalized)?;
            let subseq_encoding = self.do_tokenize(
                pre_tokenized,
                type_id,
                if is_pre_tokenized {
                    Some(subseq_idx as u32)
                } else {
                    None
                },
                offsets_type,
            )?;

            Ok(subseq_encoding)
        };

        match sequence {
            InputSequence::PreTokenized(seq) => seq
                .iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::PreTokenizedOwned(seq) => seq
                .iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::PreTokenizedCow(seq) => seq
                .iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::Raw(seq) => encode(false, 0, seq.as_ref()),
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
    pub fn encode<'s, E>(&self, input: E, add_special_tokens: bool) -> Result<Encoding, Error>
    where
        E: Into<EncodeInput<'s>>,
    {
        // Extract sequences from the EncodeInput
        let (sequence, pair) = match input.into() {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        // Encode each sequence
        let encoding = self.encode_single_sequence(sequence, 0, OffsetType::Byte)?;
        let pair_encoding = pair
            .map(|sequence| self.encode_single_sequence(sequence, 1, OffsetType::Byte))
            .transpose()?;

        // And finally post process
        self.post_process(encoding, pair_encoding, add_special_tokens)
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
    pub fn encode_char_offsets<'s, E>(
        &self,
        input: E,
        add_special_tokens: bool,
    ) -> Result<Encoding, Error>
    where
        E: Into<EncodeInput<'s>>,
    {
        // Extract sequences from the EncodeInput
        let (sequence, pair) = match input.into() {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        // Encode each sequence
        let encoding = self.encode_single_sequence(sequence, 0, OffsetType::Char)?;
        let pair_encoding = pair
            .map(|sequence| self.encode_single_sequence(sequence, 1, OffsetType::Char))
            .transpose()?;

        // And finally post process
        self.post_process(encoding, pair_encoding, add_special_tokens)
    }

    /// Decode the given ids, back to a String
    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String, Error> {
        let tokens = ids
            .into_iter()
            .filter_map(|id| {
                self.added_vocabulary
                    .id_to_token(id, &self.model)
                    .filter(|token| {
                        !skip_special_tokens || !self.added_vocabulary.is_special_token(token)
                    })
            })
            .collect::<Vec<_>>();

        if let Some(decoder) = &self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(" "))
        }
    }

    /// Tokenization logic, makes the bridge between the pre-tokenization phase and the real
    /// tokenization phase, and converting offsets back to the original referential.
    pub fn do_tokenize<P: Into<PreTokenizedString>>(
        &self,
        pretokenized: P,
        type_id: u32,
        word_idx: Option<u32>,
        offsets_type: OffsetType,
    ) -> Result<Encoding, Error> {
        let mut pretokenized: PreTokenizedString = pretokenized.into();
        pretokenized.tokenize(|normalized| self.model.tokenize(normalized.get()))?;
        pretokenized.into_encoding(word_idx, type_id, offsets_type)
    }

    /// Normalization logic, go through all normalizers
    pub fn do_normalize<V: Into<NormalizedString>>(
        &self,
        normalized: V,
    ) -> Result<NormalizedString, Error> {
        let mut normalized: NormalizedString = normalized.into();

        if let Some(ref normalizer) = self.normalizer {
            normalizer.normalize(&mut normalized)?;
        }

        Ok(normalized)
    }

    /// Register the given tokens as special tokens. This is especially useful for removing
    /// these special tokens while decoding
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        self.added_vocabulary
            .add_special_tokens(tokens, &self.model, self.normalizer.as_ref())
    }

    /// Add the given tokens to the added vocabulary
    pub fn add_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        self.added_vocabulary
            .add_tokens(tokens, &self.model, self.normalizer.as_ref())
    }

    /// PreTokenization logic, handling the case where there is no PreTokenizer set
    pub fn do_pre_tokenize<P: Into<PreTokenizedString>>(
        &self,
        pretokenized: P,
    ) -> Result<PreTokenizedString, Error> {
        let mut pretokenized: PreTokenizedString = pretokenized.into();

        if let Some(ref pretok) = self.pre_tokenizer {
            pretok.pre_tokenize(&mut pretokenized)?;
        }

        Ok(pretokenized)
    }

    /// Post processing logic, handling the case where there is no PostProcessor set
    pub fn post_process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding, Error> {
        // 1. First we truncate if needed
        let (encoding, pair_encoding) = {
            if let Some(trunc) = &self.truncation {
                let n_added_tokens = if let Some(processor) = &self.post_processor {
                    processor.added_tokens(pair_encoding.is_some())
                } else {
                    0
                };

                if add_special_tokens && n_added_tokens > 0 {
                    let params = TruncationParams {
                        max_length: trunc.max_length - n_added_tokens,
                        ..*trunc
                    };
                    truncate_encodings(encoding, pair_encoding, &params)?
                } else {
                    truncate_encodings(encoding, pair_encoding, &trunc)?
                }
            } else {
                (encoding, pair_encoding)
            }
        };

        // 2. Then We post process
        let final_encoding = if let Some(processor) = &self.post_processor {
            processor.process(encoding, pair_encoding, add_special_tokens)?
        } else {
            BertProcessing::default_process(encoding, pair_encoding, add_special_tokens)?
        };

        // 3. Then we pad if needed
        let final_encoding = if let Some(params) = &self.padding {
            params.pad_encoding(final_encoding)
        } else {
            final_encoding
        };

        Ok(final_encoding)
    }

    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, Error>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let mut encodings = inputs
            .into_iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>, Error>>()?;

        if let Some(params) = &self.padding {
            // We do the padding here to make sure we handle the batch padding
            encodings = params.pad_encodings(encodings);
        }

        Ok(encodings)
    }

    /// Encode all the sentences in parallel, using multiple threads.
    /// The offsets on each `Encoding` will be relative to chars instead of bytes.
    pub fn encode_batch_char_offsets<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, Error>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let mut encodings = inputs
            .into_iter()
            .map(|input| self.encode_char_offsets(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>, Error>>()?;

        if let Some(params) = &self.padding {
            // We do the padding here to make sure we handle the batch padding
            encodings = params.pad_encodings(encodings);
        }

        Ok(encodings)
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
