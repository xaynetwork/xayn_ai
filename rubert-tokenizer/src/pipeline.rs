use std::{borrow::Cow, collections::HashMap, iter::IntoIterator};

use anyhow::anyhow;

use crate::{
    decoder::WordPieceDecoder,
    encoding::Encoding,
    model::WordPiece,
    normalizer::{NormalizedString, Normalizer, Offsets},
    padding::Padding,
    pre_tokenizer::{BertPreTokenizer, OffsetType, PreTokenizedString},
    processor::BertProcessing,
    truncation::Truncation,
    Error,
};

pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: Offsets,
}
impl Token {
    pub fn new(id: u32, value: String, offsets: Offsets) -> Self {
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
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<BertPreTokenizer>,
    post_processor: Option<BertProcessing>,
    decoder: Option<WordPieceDecoder>,
    truncation: Truncation,
    padding: Padding,
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
            truncation: Truncation::None,
            padding: Padding::None,
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
    pub fn with_normalizer(mut self, normalizer: Option<Normalizer>) -> Self {
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
    pub fn with_truncation(mut self, trunc: Truncation) -> Self {
        self.truncation = trunc;
        self
    }

    /// Set the padding parameters.
    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }
}

pub struct TokenizerImpl {
    // Tokenizer parts
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<BertPreTokenizer>,
    model: WordPiece,
    post_processor: Option<BertProcessing>,
    decoder: Option<WordPieceDecoder>,
    // General processing parameters
    truncation: Truncation,
    padding: Padding,
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
            truncation: Truncation::None,
            padding: Padding::None,
        }
    }

    /// Set the normalizer
    pub fn with_normalizer(&mut self, normalizer: impl Into<Normalizer>) -> &mut Self {
        self.normalizer = Some(normalizer.into());
        self
    }

    /// Get the normalizer
    pub fn get_normalizer(&self) -> Option<&Normalizer> {
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
    pub fn with_truncation(&mut self, trunc: Truncation) -> &mut Self {
        self.truncation = trunc;
        self
    }

    /// Get the currently set truncation parameters
    pub fn get_truncation(&self) -> &Truncation {
        &self.truncation
    }

    /// Get a mutable reference to the currently set truncation parameters
    pub fn get_truncation_mut(&mut self) -> &mut Truncation {
        &mut self.truncation
    }

    /// Set the padding parameters
    pub fn with_padding(&mut self, padding: Padding) -> &mut Self {
        self.padding = padding;
        self
    }

    /// Get the currently set padding parameters
    pub fn get_padding(&self) -> &Padding {
        &self.padding
    }

    /// Get a mutable reference to the currently set padding parameters
    pub fn get_padding_mut(&mut self) -> &mut Padding {
        &mut self.padding
    }

    /// Get the vocabulary
    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.model.get_vocab()
    }

    /// Get the size of the vocabulary
    pub fn get_vocab_size(&self) -> usize {
        self.model.get_vocab_size()
    }

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.token_to_id(token)
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.id_to_token(id)
    }

    /// Encode a single sequence
    fn encode_single_sequence(
        &self,
        sequence: InputSequence,
        type_id: u32,
        offsets_type: OffsetType,
    ) -> Result<Encoding, Error> {
        let encode =
            |is_pre_tokenized: bool, subseq_idx: usize, subseq: &str| -> Result<Encoding, Error> {
                let pre_tokenized = self.do_pre_tokenize(subseq)?;
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
            InputSequence::PreTokenized(sequence) => sequence
                .into_iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::PreTokenizedOwned(sequence) => sequence
                .into_iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::PreTokenizedCow(sequence) => sequence
                .into_iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::Raw(sequence) => encode(false, 0, sequence.as_ref()),
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
        pretokenized.tokenize(|normalized| self.model.tokenize(normalized.normalized.as_str()))?;
        pretokenized.into_encoding(word_idx, type_id, offsets_type)
    }

    /// Normalization logic, go through all normalizers
    pub fn do_normalize(
        &self,
        normalized: impl Into<NormalizedString>,
    ) -> Result<NormalizedString, Error> {
        let normalized: NormalizedString = normalized.into();
        if let Some(ref normalizer) = self.normalizer {
            normalizer.normalize(normalized)
        } else {
            Ok(normalized)
        }
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
        let added_tokens = self
            .post_processor
            .as_ref()
            .map(|processor| processor.added_tokens(false))
            .unwrap_or_default();
        let encoding = self.truncation.truncate_encoding(encoding, added_tokens);

        // 2. Then we post-process
        let final_encoding = if let Some(ref processor) = self.post_processor {
            processor.process(encoding, pair_encoding, add_special_tokens)?
        } else {
            BertProcessing::default_process(encoding, pair_encoding, add_special_tokens)?
        };

        // 3. Then we pad if needed
        let final_encoding = self.padding.pad_encoding(final_encoding);

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
        let encodings = inputs
            .into_iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>, Error>>()?;

        // We do the padding here to make sure we handle the batch padding
        let encodings = self.padding.pad_encodings(encodings);

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
        let encodings = inputs
            .into_iter()
            .map(|input| self.encode_char_offsets(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>, Error>>()?;

        // We do the padding here to make sure we handle the batch padding
        let encodings = self.padding.pad_encodings(encodings);

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
