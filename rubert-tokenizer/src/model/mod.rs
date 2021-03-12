pub mod string;

use std::{
    collections::HashMap,
    io::{BufRead, Error as IoError},
};

use displaydoc::Display;
use num_traits::FromPrimitive;
use thiserror::Error;

use crate::{
    model::string::TokenizedString,
    pre_tokenizer::string::PreTokenizedString,
    SmallString,
};

/// A vocabulary mapping tokens to ids.
pub type Vocab<N> = HashMap<String, N>;

/// A Bert word piece model.
pub struct Model<N> {
    pub vocab: Vocab<N>,
    pub unk_id: N,
    pub unk_token: SmallString,
    pub prefix: SmallString,
    pub max_chars: usize,
}

/// The potential errors of the word piece model.
#[derive(Debug, Display, Error)]
pub enum ModelError {
    /// Overflowing output data type.
    DataType,
    /// Failed to parse the vocabulary: {0}
    Vocab(#[from] IoError),
    /// Missing the unknown token in the vocabulary
    UnkToken,
    /// Missing the continuing subword prefix in the vocabulary
    SubwordPrefix,
}

impl<N> Model<N> {
    /// Parses the vocabulary.
    pub fn parse_vocab(vocab: impl BufRead) -> Result<Vocab<N>, ModelError>
    where
        N: FromPrimitive,
    {
        vocab
            .lines()
            .enumerate()
            .map(|(idx, word)| -> Result<(String, N), ModelError> {
                Ok((
                    word?.trim().to_string(),
                    N::from_usize(idx).ok_or(ModelError::DataType)?,
                ))
            })
            .collect()
    }

    /// Creates a Bert word piece model.
    pub fn new(
        vocab: Vocab<N>,
        unk_token: SmallString,
        prefix: SmallString,
        max_chars: usize,
    ) -> Result<Self, ModelError>
    where
        N: Copy,
    {
        let unk_id = vocab
            .get(unk_token.as_str())
            .copied()
            .ok_or(ModelError::UnkToken)?;

        if !vocab.keys().any(|word| word.contains(prefix.as_str())) {
            return Err(ModelError::SubwordPrefix);
        }

        Ok(Model {
            vocab,
            unk_id,
            unk_token,
            prefix,
            max_chars,
        })
    }

    /// Tokenizes the sequences.
    pub fn tokenize(&self, sequence: PreTokenizedString) -> TokenizedString<N>
    where
        N: Copy,
    {
        TokenizedString::from(sequence).tokenize(self)
    }
}
