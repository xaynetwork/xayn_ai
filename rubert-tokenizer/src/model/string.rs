use std::borrow::Cow;

use anyhow::anyhow;

use crate::{
    model::{encoding::Encoding, Vocab},
    normalizer::{NormalizedString, Offsets, Range},
    pre_tokenizer::{BytesToCharOffsetConverter, OffsetType, PreTokenizedString},
    tokenizer::Token,
    Error,
};

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
    /// Tokens associated to this Split
    tokens: Vec<Token>,
}

impl From<NormalizedString> for Split {
    fn from(normalized: NormalizedString) -> Self {
        Self {
            normalized,
            tokens: Vec::new(),
        }
    }
}

pub struct TokenizedString {
    original: String,
    splits: Vec<Split>,
}

impl From<PreTokenizedString> for TokenizedString {
    fn from(pre_tokenized: PreTokenizedString) -> Self {
        Self {
            original: pre_tokenized.original,
            splits: pre_tokenized
                .splits
                .into_iter()
                .map(|normalized| normalized.into())
                .collect(),
        }
    }
}

impl TokenizedString {
    /// Tokenizes all the splits that do not have attached `Tokens`, using the provided function.
    pub(crate) fn tokenize(
        mut self,
        unk_token: &str,
        unk_id: u32,
        max_chars: usize,
        prefix: &str,
        vocab: &Vocab,
    ) -> Result<Self, Error> {
        for split in self.splits.iter_mut() {
            let sequence = split.normalized.normalized.as_str();
            let char_len = sequence.chars().count();
            split.tokens = if char_len > max_chars {
                vec![Token {
                    value: unk_token.to_string(),
                    id: unk_id,
                    offsets: Offsets(0, sequence.len()),
                }]
            } else {
                let mut is_bad = false;
                let mut start = 0;
                let mut sub_tokens: Vec<Token> = vec![];

                while start < sequence.len() {
                    let mut end = sequence.len();
                    let mut cur_str = None;

                    while start < end {
                        let mut substr: Cow<str> = Cow::Borrowed(&sequence[start..end]);

                        if start > 0 {
                            substr = Cow::Owned(format!("{}{}", prefix, substr));
                        }
                        if vocab.contains_key(substr.as_ref()) {
                            cur_str = Some(Token {
                                id: vocab[substr.as_ref()],
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
                    vec![Token {
                        value: unk_token.to_string(),
                        id: unk_id,
                        offsets: Offsets(0, sequence.len()),
                    }]
                } else {
                    sub_tokens
                }
            }
        }

        Ok(self)
    }

    /// Transform the current `PreTokenizedString` into an `Encoding`.
    ///
    /// If a `word_idx` is provided, any word in the generated `Encoding`
    /// will be set to this value. This is generally used with pre-tokenized
    /// input, that do not need the `PreTokenizedString` to generate word ids.
    ///
    /// This method will fail if some splits do not have associated `Token`.
    pub(crate) fn into_encoding(
        self,
        word_idx: Option<u32>,
        type_id: u32,
        offset_type: OffsetType,
    ) -> Result<Encoding, Error> {
        if self.splits.is_empty() {
            Ok(Encoding::default())
        } else if self.splits.iter().any(|split| split.tokens.is_empty()) {
            Err(anyhow!(
                "Split has not been tokenized, call `PreTokenizedString::tokenize` first"
            ))
        } else {
            let offset_converter = match offset_type {
                OffsetType::Char => Some(BytesToCharOffsetConverter::new(&self.original)),
                OffsetType::Byte => None,
            };

            Ok(self
                .splits
                .into_iter()
                .enumerate()
                .flat_map(|(idx, split)| {
                    let normalized = split.normalized;
                    let offsets = normalized.offsets_original();
                    let offset_converter = &offset_converter;

                    split.tokens.into_iter().map(move |token| {
                        let mut offsets = normalized
                            .convert_offsets(Range::Normalized(token.offsets.0..token.offsets.1))
                            .map_or(token.offsets, |range| {
                                Offsets(offsets.0 + range.start, offsets.0 + range.end)
                            });

                        // Convert to char offsets if relevant
                        if let Some(converter) = offset_converter {
                            offsets = converter.convert(offsets).unwrap_or(offsets);
                        }

                        (
                            token.id,
                            token.value,
                            offsets,
                            if word_idx.is_some() {
                                word_idx
                            } else {
                                Some(idx as u32)
                            },
                            type_id,
                        )
                    })
                })
                .collect())
        }
    }
}
