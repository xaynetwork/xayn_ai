pub mod string;

use std::{collections::HashMap, io::BufRead};

use anyhow::bail;

use crate::{model::string::TokenizedString, pre_tokenizer::string::PreTokenizedString, Error};

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

impl Model {
    /// Parses the vocabulary.
    pub fn parse_vocab(vocab: impl BufRead) -> Result<Vocab, Error> {
        vocab
            .lines()
            .enumerate()
            .map(|(idx, word)| {
                word.map(|word| (word.trim().to_string(), idx as u32))
                    .map_err(Into::into)
            })
            .collect()
    }

    /// Creates a Bert word piece model.
    pub fn new(vocab: Vocab, unk: String, prefix: String, max_chars: usize) -> Result<Self, Error> {
        let unk_id = if let Some(id) = vocab.get(unk.as_str()) {
            *id
        } else {
            bail!("padding token doesn't exist in the vocab");
        };
        if !vocab.keys().any(|word| word.contains(prefix.as_str())) {
            bail!("continuing subword prefix doesn't exist in the vocab");
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
    pub fn tokenize(&self, sequence: PreTokenizedString) -> Result<TokenizedString, Error> {
        TokenizedString::from(sequence).tokenize(self)
    }
}
