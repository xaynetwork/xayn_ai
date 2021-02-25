use std::collections::HashMap;

use anyhow::anyhow;
use unicode_categories::UnicodeCategories;

use crate::{
    encoding::Encoding,
    normalizer::{NormalizedString, OffsetReferential, Offsets, Range, SplitDelimiterBehavior},
    tokenizer::Token,
    Error,
};

/// A pre-tokenizer.
///
/// Defaults to the [`none()`] pre-tokenizer.
pub struct PreTokenizer(PreTokenizers);

/// The pre-tokenizers.
enum PreTokenizers {
    /// No pre-tokenization.
    None,
    /// Bert pre-tokenization.
    Bert,
}

impl Default for PreTokenizer {
    fn default() -> Self {
        Self::none()
    }
}

impl PreTokenizer {
    /// Creates an inert pre-tokenizer.
    pub fn none() -> Self {
        Self(PreTokenizers::None)
    }

    /// Creates a Bert pre-tokenizer.
    pub fn bert() -> Self {
        Self(PreTokenizers::Bert)
    }

    pub(crate) fn pre_tokenize(
        &self,
        normalized: impl Into<PreTokenizedString>,
    ) -> Result<PreTokenizedString, Error> {
        match self.0 {
            PreTokenizers::None => Ok(normalized.into()),
            PreTokenizers::Bert => normalized
                .into()
                .split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))?
                .split(|_, s| {
                    s.split(
                        |c: char| c.is_ascii_punctuation() || c.is_punctuation(),
                        SplitDelimiterBehavior::Isolated,
                    )
                }),
        }
    }
}

/// Various possible types of offsets
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OffsetType {
    Byte,
    Char,
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
    /// Optional Tokens associated to this Split
    tokens: Option<Vec<Token>>,
}

impl From<NormalizedString> for Split {
    fn from(normalized: NormalizedString) -> Self {
        Self {
            normalized: normalized,
            tokens: None,
        }
    }
}

impl From<(NormalizedString, Option<Vec<Token>>)> for Split {
    fn from(normalized: (NormalizedString, Option<Vec<Token>>)) -> Self {
        Self {
            normalized: normalized.0,
            tokens: normalized.1,
        }
    }
}

/// The `PreTokenizedString` is in charge of splitting an underlying string,
/// making sure everything is fine while doing so, and providing ways to normalize
/// and tokenize these splits.
/// Once everything has been normalized and tokenized, the `PreTokenizedString` is able
/// to build an `Encoding` with all the relevant offsets and word ids, relative to the
/// original string.
pub struct PreTokenizedString {
    original: String,
    splits: Vec<Split>,
}

impl PreTokenizedString {
    /// Split the `PreTokenizedString` by providing a `split_fn` in charge of splitting
    /// each substring (`NormalizedString`) into multiple parts.
    ///
    /// `split_fn` takes a `NormalizedString` and is in charge of returning an iterator
    /// over the produced `NormalizedString`. `split_fn` is free of modifying these
    /// `NormalizedString` as relevant, as long as it respects the constraint stated below.
    ///
    /// There are only one constraint that *MUST* be respected:
    /// > The produced `NormalizedString`, if combined back together, must have the
    /// same `original` string as the original one given to `split_fn`. This concretely
    /// means that for the offset tracking to work as expected, `split_fn` must produce
    /// "splits" of the original string.
    fn split<F, U, R>(mut self, mut split_fn: F) -> Result<Self, Error>
    where
        F: FnMut(usize, NormalizedString) -> Result<U, Error>,
        U: IntoIterator<Item = R>,
        R: Into<Split>,
    {
        // new_splits is at least as big as self.splits
        let mut new_splits = Vec::with_capacity(self.splits.len());
        for (i, original_split) in self.splits.drain(..).enumerate() {
            if original_split.tokens.is_some() {
                new_splits.push(original_split);
                continue;
            }

            new_splits.extend(
                split_fn(i, original_split.normalized)?
                    .into_iter()
                    .filter_map(|split| {
                        let split: Split = split.into();
                        if split.normalized.normalized.is_empty() {
                            None
                        } else {
                            Some(split)
                        }
                    }),
            );
        }
        self.splits = new_splits;

        Ok(self)
    }

    /// Normalized all the splits that do not have attached `Tokens`, using the provided
    /// `normalize` function.
    fn normalize<F>(mut self, normalize: F) -> Result<Self, Error>
    where
        F: Fn(&NormalizedString) -> Result<NormalizedString, Error>,
    {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            split.normalized = normalize(&split.normalized)?;
        }

        Ok(self)
    }

    /// Tokenizes all the splits that do not have attached `Tokens`, using the provided function.
    pub(crate) fn tokenize(
        mut self,
        f: impl Fn(&NormalizedString) -> Result<Vec<Token>, Error>,
    ) -> Result<Self, Error> {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            split.tokens = Some(f(&split.normalized)?);
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
        } else if !self.splits.iter().all(|split| split.tokens.is_some()) {
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

                    split.tokens.unwrap().into_iter().map(move |token| {
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

    /// Returns a list of splits, each of them being a slice of the normalized
    /// string, the associated offsets either in original or normalized
    /// referential, as well as the potention tokens
    fn get_splits(
        &self,
        offset_ref: OffsetReferential,
        offset_type: OffsetType,
    ) -> Vec<(&str, Offsets, &Option<Vec<Token>>)> {
        let offset_converter = match offset_type {
            OffsetType::Char => Some(BytesToCharOffsetConverter::new(&self.original)),
            OffsetType::Byte => None,
        };

        let mut offset = 0;
        self.splits
            .iter()
            .map(|split| {
                let mut offsets = match offset_ref {
                    OffsetReferential::Original => split.normalized.offsets_original(),
                    OffsetReferential::Normalized => {
                        let len = split.normalized.normalized.len();
                        offset += len;
                        Offsets(offset - len, offset)
                    }
                };

                // Convert to char offsets if relevant
                if let Some(ref converter) = offset_converter {
                    offsets = converter.convert(offsets).unwrap_or(offsets);
                }

                (split.normalized.normalized.as_str(), offsets, &split.tokens)
            })
            .collect()
    }
}

impl From<NormalizedString> for PreTokenizedString {
    fn from(normalized: NormalizedString) -> Self {
        Self {
            original: normalized.original.clone(),
            splits: vec![Split {
                normalized: normalized,
                tokens: None,
            }],
        }
    }
}

impl From<&str> for PreTokenizedString {
    fn from(string: &str) -> Self {
        NormalizedString::from(string).into()
    }
}

impl From<String> for PreTokenizedString {
    fn from(string: String) -> Self {
        NormalizedString::from(string).into()
    }
}

struct BytesToCharOffsetConverter {
    map: HashMap<usize, usize>,
}

impl BytesToCharOffsetConverter {
    pub fn new(sequence: &str) -> Self {
        Self {
            map: sequence
                .char_indices()
                .enumerate()
                .flat_map(|(i, (b, c))| {
                    let mut n = 0;
                    std::iter::repeat_with(move || {
                        let o = (b + n, i);
                        n += 1;
                        o
                    })
                    .take(c.len_utf8())
                })
                .collect(),
        }
    }

    fn convert(&self, offsets: Offsets) -> Option<Offsets> {
        match (self.map.get(&offsets.0), self.map.get(&offsets.1)) {
            (Some(start), Some(end)) => Some(Offsets(*start, *end)),
            // If we reached the end, `end` is not in the map
            (Some(start), None) => {
                // But the one just before should be
                let last = self.map.get(&(offsets.1 - 1)).copied().unwrap_or(start + 1);
                Some(Offsets(*start, last + 1))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let pretokenized = PreTokenizer::bert()
            .pre_tokenize("Hey friend!     How are you?!?")
            .unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hey", Offsets(0, 3)),
                ("friend", Offsets(4, 10)),
                ("!", Offsets(10, 11)),
                ("How", Offsets(16, 19)),
                ("are", Offsets(20, 23)),
                ("you", Offsets(24, 27)),
                ("?", Offsets(27, 28)),
                ("!", Offsets(28, 29)),
                ("?", Offsets(29, 30)),
            ],
        );
    }

    #[test]
    fn chinese_chars() {
        let sequence = "野口里佳 Noguchi Rika";
        let normalized = NormalizedString::from(sequence).transform(
            sequence.chars().flat_map(|c| {
                if (c as usize) > 0x4E00 {
                    vec![(' ', 0), (c, 1), (' ', 1)]
                } else {
                    vec![(c, 0)]
                }
            }),
            0,
        );
        let pretokenized = PreTokenizer::bert().pre_tokenize(normalized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("野", Offsets(0, 3)),
                ("口", Offsets(3, 6)),
                ("里", Offsets(6, 9)),
                ("佳", Offsets(9, 12)),
                ("Noguchi", Offsets(13, 20)),
                ("Rika", Offsets(21, 25)),
            ],
        );
    }
}
