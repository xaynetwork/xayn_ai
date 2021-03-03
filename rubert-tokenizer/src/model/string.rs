use std::borrow::Cow;

use crate::{
    model::Model,
    normalizer::string::{NormalizedString, Offsets},
    pre_tokenizer::string::PreTokenizedString,
};

/// A token relative to a sequence.
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: Offsets,
}

/// A subpart of a normalized string.
pub struct Split {
    pub normalized: NormalizedString,
    pub tokens: Vec<Token>,
}

impl From<NormalizedString> for Split {
    fn from(string: NormalizedString) -> Self {
        Self {
            normalized: string,
            tokens: Vec::new(),
        }
    }
}

/// A tokenized sequence.
pub struct TokenizedString {
    pub splits: Vec<Split>,
}

impl From<PreTokenizedString> for TokenizedString {
    fn from(string: PreTokenizedString) -> Self {
        Self {
            splits: string.splits.into_iter().map(Into::into).collect(),
        }
    }
}

impl TokenizedString {
    /// Tokenizes wrt the model parameters.
    pub fn tokenize(mut self, model: &Model) -> Self {
        self.splits.iter_mut().for_each(|split| {
            let string = split.normalized.normalized.as_str();
            let len = string.len();
            if string.chars().count() > model.max_chars {
                split.tokens = vec![Token {
                    id: model.unk_id,
                    value: model.unk_token.clone(),
                    offsets: Offsets(0, len),
                }]
            } else {
                let mut start = 0;
                while start < len {
                    let mut end = len;
                    start = loop {
                        if start >= end {
                            split.tokens = vec![Token {
                                id: model.unk_id,
                                value: model.unk_token.clone(),
                                offsets: Offsets(0, len),
                            }];
                            return;
                        }

                        let sub_str = if start > 0 {
                            Cow::Owned([model.prefix.as_str(), &string[start..end]].join(""))
                        } else {
                            Cow::Borrowed(&string[start..end])
                        };

                        if let Some(id) = model.vocab.get(sub_str.as_ref()) {
                            split.tokens.push(Token {
                                id: *id,
                                value: sub_str.into_owned(),
                                offsets: Offsets(start, end),
                            });
                            break end;
                        } else {
                            end -= sub_str.chars().last().map_or(1, |c| c.len_utf8());
                        }
                    }
                }
            };
        });

        self
    }
}
