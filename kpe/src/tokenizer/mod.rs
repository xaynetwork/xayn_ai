pub mod encoding;
pub mod key_phrase;

use std::io::BufRead;

use displaydoc::Display;
use rubert_tokenizer::{Builder, BuilderError, Padding, Tokenizer as BertTokenizer, Truncation};
use thiserror::Error;

/// A pre-configured Bert tokenizer for key phrase extraction.
#[derive(Debug)]
pub struct Tokenizer {
    tokenizer: BertTokenizer<i64>,
    token_size: usize,
    key_phrase_size: usize,
}

/// The potential errors of the tokenizer.
#[derive(Debug, Display, Error, PartialEq)]
pub enum TokenizerError {
    /// Failed to build the tokenizer: {0}
    Builder(#[from] BuilderError),
}

impl Tokenizer {
    /// Creates a tokenizer from a vocabulary.
    ///
    /// Can be set to keep accents and to lowercase the sequences. Requires the maximum number of
    /// tokens per tokenized sequence, which applies to padding and truncation and includes special
    /// tokens as well. Also requires the maximum number of words per key phrase.
    pub fn new(
        vocab: impl BufRead,
        accents: bool,
        lowercase: bool,
        token_size: usize,
        key_phrase_size: usize,
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
            token_size,
            key_phrase_size,
        })
    }
}
