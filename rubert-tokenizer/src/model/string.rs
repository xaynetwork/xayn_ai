use std::{borrow::Cow, convert::TryFrom};

use anyhow::anyhow;

use crate::{
    model::{encoding::Encoding, Model},
    normalizer::string::{NormalizedString, Offsets, Range},
    pre_tokenizer::string::PreTokenizedString,
    Error,
};

pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: Offsets,
}

/// Wrapper for a subpart of a `NormalizedString`.
///
/// This Split contains the underlying `NormalizedString` as well as its offsets
/// in the original string. These offsets are in the `original` referential.
/// It also contains any `Token` associated to the current split
pub struct Split {
    /// The underlying `NormalizedString`. Each SubString is represented by a `NormalizedString`
    /// and in the end we might be carrying a lot of SubString representing various parts of the
    /// original input string.
    normalized: NormalizedString,
    /// Tokens associated to this Split.
    tokens: Vec<Token>,
}

impl From<NormalizedString> for Split {
    fn from(string: NormalizedString) -> Self {
        Self {
            normalized: string,
            tokens: Vec::new(),
        }
    }
}

pub struct TokenizedString {
    splits: Vec<Split>,
}

impl From<PreTokenizedString> for TokenizedString {
    fn from(string: PreTokenizedString) -> Self {
        Self {
            splits: string.splits.into_iter().map(Into::into).collect(),
        }
    }
}

impl TokenizedString {
    pub fn tokenize(mut self, model: &Model) -> Result<Self, Error> {
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

        Ok(self)
    }
}

impl TryFrom<TokenizedString> for Encoding {
    type Error = Error;

    fn try_from(string: TokenizedString) -> Result<Self, Self::Error> {
        if string.splits.is_empty() {
            Ok(Encoding::default())
        } else if string.splits.iter().any(|split| split.tokens.is_empty()) {
            Err(anyhow!(
                "Split has not been tokenized, call `PreTokenizedString::tokenize` first"
            ))
        } else {
            Ok(string
                .splits
                .into_iter()
                .enumerate()
                .flat_map(|(idx, split)| {
                    let Split { normalized, tokens } = split;
                    tokens.into_iter().map(move |mut token| {
                        token.offsets = normalized
                            .convert_offsets(Range::Normalized(token.offsets.0..token.offsets.1))
                            .map_or(token.offsets, |range| {
                                Offsets(
                                    normalized.original_shift + range.start,
                                    normalized.original_shift + range.end,
                                )
                            });
                        (token, Some(idx as u32))
                    })
                })
                .collect())
        }
    }
}
