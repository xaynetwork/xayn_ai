use std::{borrow::Cow, collections::HashMap, io::BufRead};

use anyhow::bail;

use crate::{encoding::Encoding, normalizer::Offsets, tokenizer::Token, Error};

pub(crate) type Vocab = HashMap<String, u32>;

/// A Bert word piece model.
pub struct Model {
    pub vocab: Vocab,
    pub unk_id: u32,
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
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
            .any(|word| word.contains(self.continuing_subword_prefix.as_str()))
        {
            bail!("continuing subword prefix doesn't exist in the vocab");
        }
        Ok(self)
    }

    pub fn tokenize(&self, sequence: &str) -> Result<Vec<Token>, Error> {
        let char_len = sequence.chars().count();
        if char_len > self.max_input_chars_per_word {
            return Ok(vec![Token {
                value: self.unk_token.clone(),
                id: self.unk_id,
                offsets: Offsets(0, sequence.len()),
            }]);
        }

        let mut is_bad = false;
        let mut start = 0;
        let mut sub_tokens: Vec<Token> = vec![];

        while start < sequence.len() {
            let mut end = sequence.len();
            let mut cur_str = None;

            while start < end {
                let mut substr: Cow<str> = Cow::Borrowed(&sequence[start..end]);

                if start > 0 {
                    substr = Cow::Owned(format!("{}{}", self.continuing_subword_prefix, substr));
                }
                if self.vocab.contains_key(substr.as_ref()) {
                    cur_str = Some(Token {
                        id: self.vocab[substr.as_ref()],
                        value: substr.to_string(),
                        offsets: Offsets(start, end),
                    });
                    break;
                }
                end -= substr.chars().last().map_or(1, |c| c.len_utf8());
            }

            if cur_str.is_none() {
                is_bad = true;
                break;
            }

            sub_tokens.push(cur_str.unwrap());
            start = end;
        }

        if is_bad {
            Ok(vec![Token {
                value: self.unk_token.clone(),
                id: self.unk_id,
                offsets: Offsets(0, sequence.len()),
            }])
        } else {
            Ok(sub_tokens)
        }
    }

    pub fn de_tokenize(&self, encoding: &Encoding, cleanup: bool) -> String {
        let unk = self.unk_token.as_str();
        let tokens = encoding
            .tokens
            .iter()
            .filter_map(|token| {
                if !cleanup || token != unk {
                    Some(token.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut string = tokens.join(" ").replace(
            format!(" {}", self.continuing_subword_prefix.as_str()).as_str(),
            "",
        );
        if cleanup {
            string = string
                .replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" do not", " don't")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re");
        }

        string
    }
}
