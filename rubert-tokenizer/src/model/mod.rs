pub mod string;

use std::{
    collections::HashMap,
    io::{BufRead, Error as IoError},
};

use displaydoc::Display;
use thiserror::Error;

use crate::{model::string::TokenizedString, pre_tokenizer::string::PreTokenizedString};

/// A vocabulary mapping tokens to ids.
pub type Vocab = HashMap<String, u32>;

/// A Bert word piece model.
pub struct Model {
    pub vocab: Vocab,
    pub unk_id: u32,
    pub unk_token: String,
    pub prefix: String,
    pub max_chars: usize,
}

/// The potential errors of the word piece model.
#[derive(Debug, Display, Error)]
pub enum ModelError {
    /// Failed to parse the vocabulary: {0}
    Vocab(#[from] IoError),
    /// Missing the unknown token in the vocabulary
    UnkToken,
    /// Missing the continuing subword prefix in the vocabulary
    SubwordPrefix,
}

impl Model {
    /// Parses the vocabulary.
    pub fn parse_vocab(vocab: impl BufRead) -> Result<Vocab, ModelError> {
        vocab
            .lines()
            .enumerate()
            .map(|(idx, word)| word.map(|word| (word.trim().to_string(), idx as u32)))
            .collect::<Result<_, _>>()
            .map_err(Into::into)
    }

    /// Creates a Bert word piece model.
    pub fn new(
        vocab: Vocab,
        unk: String,
        prefix: String,
        max_chars: usize,
    ) -> Result<Self, ModelError> {
        let unk_id = vocab
            .get(unk.as_str())
            .copied()
            .ok_or(ModelError::UnkToken)?;
        if !vocab.keys().any(|word| word.contains(prefix.as_str())) {
            return Err(ModelError::SubwordPrefix);
        }

        Ok(Model {
            vocab,
            unk_id,
            unk_token: unk,
            prefix,
            max_chars,
        })
    }

    /// Tokenizes the sequences.
    pub fn tokenize(&self, sequence: PreTokenizedString) -> TokenizedString {
        TokenizedString::from(sequence).tokenize(self)
    }
}
