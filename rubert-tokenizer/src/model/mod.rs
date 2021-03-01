mod encoding;
mod string;

use std::{collections::HashMap, io::BufRead};

use anyhow::bail;

pub use self::{encoding::Encoding, string::TokenizedString};
use crate::{pre_tokenizer::PreTokenizedString, Error};

pub(crate) type Vocab = HashMap<String, u32>;

/// A Bert word piece model.
pub struct Model {
    pub(crate) vocab: Vocab,
    pub(crate) unk_id: u32,
    pub(crate) unk_token: String,
    /// Continuing subword prefix.
    pub(crate) prefix: String,
    /// Maximum input characters per word.
    pub(crate) max_chars: usize,
}

impl Model {
    /// Parses the in-memory vocabulary.
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

    /// Validates itself.
    pub fn validate(mut self) -> Result<Self, Error> {
        if let Some(id) = self.vocab.get(self.unk_token.as_str()) {
            self.unk_id = *id;
        } else {
            bail!("padding token doesn't exist in the vocab");
        }
        if !self
            .vocab
            .keys()
            .any(|word| word.contains(self.prefix.as_str()))
        {
            bail!("continuing subword prefix doesn't exist in the vocab");
        }
        Ok(self)
    }

    pub fn tokenize(&self, pre_tokenized: PreTokenizedString) -> Result<TokenizedString, Error> {
        TokenizedString::from(pre_tokenized).tokenize(
            self.unk_token.as_str(),
            self.unk_id,
            self.max_chars,
            self.prefix.as_str(),
            &self.vocab,
        )
    }
}
