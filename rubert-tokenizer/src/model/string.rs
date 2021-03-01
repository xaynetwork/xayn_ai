use std::borrow::Cow;

use anyhow::anyhow;

use crate::{
    model::{encoding::Encoding, Model},
    normalizer::string::{NormalizedString, Offsets, Range},
    pre_tokenizer::string::{BytesToCharOffsetConverter, OffsetType, PreTokenizedString},
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
    original: String,
    splits: Vec<Split>,
}

impl From<PreTokenizedString> for TokenizedString {
    fn from(string: PreTokenizedString) -> Self {
        Self {
            original: string.original,
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

    /// Transform the current `PreTokenizedString` into an `Encoding`.
    ///
    /// If a `word_idx` is provided, any word in the generated `Encoding`
    /// will be set to this value. This is generally used with pre-tokenized
    /// input, that do not need the `PreTokenizedString` to generate word ids.
    ///
    /// This method will fail if some splits do not have associated `Token`.
    pub fn into_encoding(self, offset_type: OffsetType) -> Result<Encoding, Error> {
        if self.splits.is_empty() {
            Ok(Encoding::default())
        } else if self.splits.iter().any(|split| split.tokens.is_empty()) {
            Err(anyhow!(
                "Split has not been tokenized, call `PreTokenizedString::tokenize` first"
            ))
        } else {
            let offset_converter = match offset_type {
                OffsetType::Byte => None,
                OffsetType::Char => Some(BytesToCharOffsetConverter::new(self.original.as_str())),
            };

            Ok(self
                .splits
                .into_iter()
                .enumerate()
                .flat_map(|(idx, split)| {
                    let Split { normalized, tokens } = split;
                    let offset_converter = offset_converter.as_ref();

                    tokens.into_iter().map(move |mut token| {
                        token.offsets = normalized
                            .convert_offsets(Range::Normalized(token.offsets.0..token.offsets.1))
                            .map_or(token.offsets, |range| {
                                Offsets(
                                    normalized.original_shift + range.start,
                                    normalized.original_shift + range.end,
                                )
                            });

                        // Convert to char offsets if relevant
                        if let Some(converter) = offset_converter {
                            token.offsets =
                                converter.convert(token.offsets).unwrap_or(token.offsets);
                        }

                        (token, Some(idx as u32))
                    })
                })
                .collect())
        }
    }
}
