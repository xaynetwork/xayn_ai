pub mod encoding;
pub mod key_phrase;

use std::io::BufRead;

use displaydoc::Display;
use rubert_tokenizer::{Builder, BuilderError, Padding, Tokenizer as BertTokenizer, Truncation};
use thiserror::Error;

/// A pre-configured Bert tokenizer for key phrase extraction.
#[derive(Debug)]
pub struct Tokenizer<const KEY_PHRASE_SIZE: usize> {
    tokenizer: BertTokenizer<i64>,
    key_phrase_max_count: Option<usize>,
    key_phrase_min_score: Option<f32>,
}

/// The potential errors of the tokenizer.
#[derive(Debug, Display, Error, PartialEq)]
pub enum TokenizerError {
    /// Failed to build the tokenizer: {0}
    Builder(#[from] BuilderError),
}

impl<const KEY_PHRASE_SIZE: usize> Tokenizer<KEY_PHRASE_SIZE> {
    /// Creates a tokenizer from a vocabulary.
    ///
    /// Can be set to keep accents and to lowercase the sequences. Requires the maximum number of
    /// tokens per tokenized sequence, which applies to padding and truncation and includes special
    /// tokens as well.
    ///
    /// Optionally takes an upper count for the number of returned key phrases as well as a lower
    /// threshold for the scores of returned key phrases.
    pub fn new(
        vocab: impl BufRead,
        accents: bool,
        lowercase: bool,
        token_size: usize,
        key_phrase_max_count: Option<usize>,
        key_phrase_min_score: Option<f32>,
    ) -> Result<Self, TokenizerError> {
        let tokenizer = Builder::new(vocab)?
            .with_normalizer(true, false, accents, lowercase)
            .with_model("[UNK]", "##", 100)
            .with_post_tokenizer("[CLS]", "[SEP]")
            .with_truncation(Truncation::fixed(token_size, 0))
            .with_padding(Padding::fixed(token_size, "[PAD]"))
            .build()?;

        Ok(Tokenizer {
            tokenizer,
            key_phrase_max_count,
            key_phrase_min_score,
        })
    }
}
