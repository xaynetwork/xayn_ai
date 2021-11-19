use std::{collections::HashMap, iter, ops::Range as StdRange};

use num_traits::{FromPrimitive, Num};

use crate::{
    model::string::{Split, TokenizedString},
    normalizer::string::{Offsets, Range},
};

/// An encoded sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoding<N> {
    /// The IDs of the tokens.
    pub(crate) ids: Vec<N>,
    /// The type of the IDs.
    pub(crate) type_ids: Vec<N>,
    /// The tokenized sequence.
    pub(crate) tokens: Vec<String>,
    /// The indices of the word associated to the tokens.
    pub(crate) word_indices: Vec<Option<N>>,
    /// The offsets of the tokens in the sequence.
    pub(crate) offsets: Vec<Offsets>,
    /// The mask identifying special tokens.
    pub(crate) special_tokens_mask: Vec<N>,
    /// The mask identifying padding tokens.
    pub(crate) attention_mask: Vec<N>,
    /// The ranges of tokens covered by each sequence. If this is None or empty, it is considered as
    /// exactly one sequence covering the entire range.
    pub(crate) sequence_ranges: Option<HashMap<usize, StdRange<usize>>>,
    /// A list of overflowing encodings produced by truncation.
    pub(crate) overflowing: Option<Vec<Encoding<N>>>,
}

impl<N> std::iter::FromIterator<Encoding<N>> for Encoding<N>
where
    N: Copy,
{
    fn from_iter<I: IntoIterator<Item = Encoding<N>>>(iter: I) -> Self {
        Self::merge(iter, false)
    }
}

impl<N> From<TokenizedString<N>> for Encoding<N>
where
    N: Num + FromPrimitive + Copy,
{
    /// Creates an encoding from a tokenized sequence.
    ///
    /// # Panics
    /// Panics if the sequence has not been tokenized before or if the token indices overflow `N`.
    fn from(sequence: TokenizedString<N>) -> Self {
        if sequence.splits.is_empty() {
            return Encoding::with_capacity(0);
        }
        assert!(
            sequence.splits.iter().all(|split| !split.tokens.is_empty()),
            "Sequence has not been tokenized, call `Model::tokenize` first",
        );

        let len = sequence
            .splits
            .iter()
            .flat_map(|split| split.tokens.iter())
            .count();
        let ids = sequence
            .splits
            .iter()
            .flat_map(|split| split.tokens.iter().map(|token| token.id))
            .collect();
        let word_indices = sequence
            .splits
            .iter()
            .enumerate()
            .flat_map(|(idx, split)| {
                iter::repeat(Some(N::from_usize(idx).unwrap())).take(split.tokens.len())
            })
            .collect();
        let offsets = sequence
            .splits
            .iter()
            .flat_map(|split| {
                let Split { normalized, tokens } = split;
                tokens.iter().map(move |token| {
                    Range::Normalized(token.offsets.0..token.offsets.1)
                        .convert(normalized)
                        .map_or(token.offsets, |range| {
                            Offsets(
                                normalized.offset + range.start,
                                normalized.offset + range.end,
                            )
                        })
                })
            })
            .collect();
        let tokens = sequence
            .splits
            .into_iter()
            .flat_map(|split| split.tokens.into_iter().map(|token| token.value))
            .collect();

        Self {
            ids,
            type_ids: vec![N::zero(); len],
            tokens,
            word_indices,
            offsets,
            special_tokens_mask: vec![N::zero(); len],
            attention_mask: vec![N::one(); len],
            sequence_ranges: None,
            overflowing: None,
        }
    }
}

impl<N> From<Encoding<N>>
    for (
        Vec<N>,
        Vec<N>,
        Vec<String>,
        Vec<Option<N>>,
        Vec<Offsets>,
        Vec<N>,
        Vec<N>,
        Option<Vec<Encoding<N>>>,
    )
{
    fn from(encoding: Encoding<N>) -> Self {
        (
            encoding.ids,
            encoding.type_ids,
            encoding.tokens,
            encoding.word_indices,
            encoding.offsets,
            encoding.special_tokens_mask,
            encoding.attention_mask,
            encoding.overflowing,
        )
    }
}

impl<N> Encoding<N> {
    /// Creates an empty encoding with capacity.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            ids: Vec::with_capacity(capacity),
            type_ids: Vec::with_capacity(capacity),
            tokens: Vec::with_capacity(capacity),
            word_indices: Vec::with_capacity(capacity),
            offsets: Vec::with_capacity(capacity),
            special_tokens_mask: Vec::with_capacity(capacity),
            attention_mask: Vec::with_capacity(capacity),
            sequence_ranges: None,
            overflowing: None,
        }
    }

    /// Gets the total length.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Checks whether this is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Gets the number of combined sequences.
    pub fn sequences_len(&self) -> usize {
        self.sequence_ranges
            .as_ref()
            .map(|sequence_ranges| sequence_ranges.len())
            .unwrap_or(1)
    }

    /// Gets the ids.
    pub fn ids(&self) -> &[N] {
        self.ids.as_slice()
    }

    /// Gets the type ids.
    pub fn type_ids(&self) -> &[N] {
        self.type_ids.as_slice()
    }

    /// Gets the tokens.
    pub fn tokens(&self) -> &[String] {
        self.tokens.as_slice()
    }

    /// Gets the word indices.
    pub fn word_indices(&self) -> &[Option<N>] {
        self.word_indices.as_slice()
    }

    /// Gets the offsets.
    pub fn offsets(&self) -> &[Offsets] {
        self.offsets.as_slice()
    }

    /// Gets the special tokens mask.
    pub fn special_tokens_mask(&self) -> &[N] {
        self.special_tokens_mask.as_slice()
    }

    /// Gets the attention mask.
    pub fn attention_mask(&self) -> &[N] {
        self.attention_mask.as_slice()
    }

    /// Gets the overflowing parts.
    pub fn overflowing(&self) -> Option<&[Encoding<N>]> {
        self.overflowing.as_deref()
    }

    /// Merges with another encoding.
    pub fn merge_with(mut self, other: Encoding<N>, growing_offsets: bool) -> Self
    where
        N: Copy,
    {
        // Handle merging the overflowing parts too: Combine them all
        // In most of the cases, we expect `other.overflowing.len() == 0`
        let mut overflowings = vec![];

        // 1. All our overflowings with all the others
        if let Some(overflowing) = self.overflowing.as_ref() {
            for this_of in overflowing {
                // 1. The other itself
                overflowings.push(this_of.clone().merge_with(other.clone(), growing_offsets));
                // 2. Its overflowings (this should rarely happen...)
                if let Some(overflowing) = other.overflowing.as_ref() {
                    for other_of in overflowing {
                        overflowings.push(
                            this_of
                                .clone()
                                .merge_with(other_of.clone(), growing_offsets),
                        );
                    }
                }
            }
        }

        // 2. Ourself with all the other overflowings (this should rarely happen too...)
        if let Some(overflowing) = other.overflowing.as_ref() {
            for other_of in overflowing {
                overflowings.push(self.clone().merge_with(other_of.clone(), growing_offsets));
            }
        }

        if !overflowings.is_empty() {
            self.overflowing = Some(overflowings);
        }

        // Finish by merging ourself with the other encoding
        let len = self.len();
        let o_len = other.len();
        match (self.sequence_ranges.as_mut(), other.sequence_ranges) {
            (Some(sequence_ranges), Some(o_sequence_ranges)) => {
                let max_seq_id = sequence_ranges.keys().max().copied().unwrap_or_default();
                sequence_ranges.extend(o_sequence_ranges.into_iter().map(|(seq_id, range)| {
                    (max_seq_id + 1 + seq_id, len + range.start..len + range.end)
                }));
            }
            (Some(sequence_ranges), None) => {
                let max_seq_id = sequence_ranges.keys().max().copied().unwrap_or_default();
                sequence_ranges.extend(iter::once((max_seq_id + 1, len..len + o_len)));
            }
            (None, Some(o_sequence_ranges)) => {
                self.sequence_ranges = Some(
                    iter::once((0, 0..len))
                        .chain(o_sequence_ranges.into_iter().map(|(seq_id, range)| {
                            (1 + seq_id, len + range.start..len + range.end)
                        }))
                        .collect(),
                );
            }
            (None, None) => {
                self.sequence_ranges = Some(
                    iter::once((0, 0..len))
                        .chain(iter::once((1, len..len + o_len)))
                        .collect(),
                );
            }
        }
        self.ids.extend(other.ids);
        self.type_ids.extend(other.type_ids);
        self.tokens.extend(other.tokens);
        self.word_indices.extend(other.word_indices);

        let starting_offset = if growing_offsets {
            self.offsets.last().map_or(0, |o| o.1)
        } else {
            0
        };
        self.offsets.extend(
            other
                .offsets
                .into_iter()
                .map(|Offsets(start, end)| Offsets(start + starting_offset, end + starting_offset))
                .collect::<Vec<_>>(),
        );
        self.special_tokens_mask.extend(other.special_tokens_mask);
        self.attention_mask.extend(other.attention_mask);

        self
    }

    /// Merges the encodings.
    pub fn merge(encodings: impl IntoIterator<Item = Encoding<N>>, growing_offsets: bool) -> Self
    where
        N: Copy,
    {
        encodings
            .into_iter()
            .fold(Encoding::with_capacity(0), |encoding, other| {
                encoding.merge_with(other, growing_offsets)
            })
    }

    /// Extends from the "slice" `&from[skip..skip+take]`.
    #[inline]
    fn extend(&mut self, from: &Self, skip: usize, take: usize)
    where
        N: Copy,
    {
        debug_assert!(self.overflowing.is_none());
        debug_assert!(self.sequence_ranges.is_none());

        self.ids.extend(from.ids.iter().skip(skip).take(take));
        self.type_ids
            .extend(from.type_ids.iter().skip(skip).take(take));
        self.tokens
            .extend(from.tokens.iter().skip(skip).take(take).cloned());
        self.word_indices
            .extend(from.word_indices.iter().skip(skip).take(take));
        self.offsets
            .extend(from.offsets.iter().skip(skip).take(take));
        self.special_tokens_mask
            .extend(from.special_tokens_mask.iter().skip(skip).take(take));
        self.attention_mask
            .extend(from.attention_mask.iter().skip(skip).take(take));
    }

    /// Drains and chains the drainage with the "slice" `&from[skip..skip+take]`.
    #[inline]
    fn drain_chain(&mut self, from: &Self, skip: usize, take: usize) -> Self
    where
        N: Copy,
    {
        debug_assert!(self.overflowing.is_none());
        debug_assert!(self.sequence_ranges.is_none());

        Self {
            ids: self
                .ids
                .drain(..)
                .chain(from.ids.iter().skip(skip).take(take).copied())
                .collect(),
            type_ids: self
                .type_ids
                .drain(..)
                .chain(from.type_ids.iter().skip(skip).take(take).copied())
                .collect(),
            tokens: self
                .tokens
                .drain(..)
                .chain(from.tokens.iter().skip(skip).take(take).cloned())
                .collect(),
            word_indices: self
                .word_indices
                .drain(..)
                .chain(from.word_indices.iter().skip(skip).take(take).copied())
                .collect(),
            offsets: self
                .offsets
                .drain(..)
                .chain(from.offsets.iter().skip(skip).take(take).copied())
                .collect(),
            special_tokens_mask: self
                .special_tokens_mask
                .drain(..)
                .chain(
                    from.special_tokens_mask
                        .iter()
                        .skip(skip)
                        .take(take)
                        .copied(),
                )
                .collect(),
            attention_mask: self
                .attention_mask
                .drain(..)
                .chain(from.attention_mask.iter().skip(skip).take(take).copied())
                .collect(),
            sequence_ranges: None,
            overflowing: None,
        }
    }

    /// Truncates to a maximum length.
    ///
    /// Truncated parts are pushed to the overflowings and are overlapping by a stride.
    ///
    /// # Panics
    /// Panics if `stride >= len` for `len != 0`.
    pub(crate) fn truncate(mut self, len: usize, stride: usize) -> Self
    where
        N: Copy,
    {
        if len >= self.len() {
            return self;
        }
        if len == 0 {
            let mut encoding = Self::with_capacity(0);
            encoding.overflowing = Some(vec![self]);
            return encoding;
        }
        assert!(
            stride < len,
            "stride must be less than the truncation length",
        );

        // Get the main overflowing part
        let overflowings = &Self {
            ids: self.ids.split_off(len),
            type_ids: self.type_ids.split_off(len),
            tokens: self.tokens.split_off(len),
            word_indices: self.word_indices.split_off(len),
            offsets: self.offsets.split_off(len),
            special_tokens_mask: self.special_tokens_mask.split_off(len),
            attention_mask: self.attention_mask.split_off(len),
            sequence_ranges: None,
            overflowing: None,
        };

        // When truncating, we lose the `sequence_ranges` information.
        self.sequence_ranges = None;

        // Now we need to separate the overflowing part into as many Encoding as needed
        let part_size = len - stride;
        self.overflowing = Some(
            (0..overflowings.ids.len())
                .step_by(part_size)
                .scan(
                    {
                        let mut initial = Self::with_capacity(stride);
                        initial.extend(&self, part_size, stride);
                        initial
                    },
                    |previous: &mut Self, part_id| {
                        let overflowing = previous.drain_chain(overflowings, part_id, part_size);
                        previous.extend(&overflowing, part_size, stride);
                        Some(overflowing)
                    },
                )
                .collect(),
        );

        self
    }

    /// Pads to a minimum length.
    pub(crate) fn pad(mut self, len: usize, pad_id: N, pad_type_id: N, pad_token: &str) -> Self
    where
        N: Num + Copy,
    {
        // Dispatch call to all the overflowings first
        self.overflowing = self.overflowing.map(|overflowing| {
            overflowing
                .into_iter()
                .map(|encoding| encoding.pad(len, pad_id, pad_type_id, pad_token))
                .collect()
        });

        // Then check if we should pad ourself
        if self.len() >= len {
            // We just do nothing if the wanted padding length is smaller than us
            return self;
        }
        let pad_length = len - self.len();

        self.ids.extend(iter::repeat(pad_id).take(pad_length));
        self.type_ids
            .extend(iter::repeat(pad_type_id).take(pad_length));
        self.tokens
            .extend(iter::repeat(pad_token.to_string()).take(pad_length));
        self.word_indices
            .extend(iter::repeat(None).take(pad_length));
        self.attention_mask
            .extend(iter::repeat(N::zero()).take(pad_length));
        self.special_tokens_mask
            .extend(iter::repeat(N::one()).take(pad_length));
        self.offsets.extend(
            iter::repeat(
                self.offsets
                    .last()
                    .map(|&Offsets(_, end)| Offsets(end, end))
                    .unwrap_or_default(),
            )
            .take(pad_length),
        );

        self
    }

    /// Decodes with optional cleanup.
    pub(crate) fn decode(
        &self,
        cls: &str,
        sep: &str,
        pad: &str,
        unk: &str,
        prefix: &str,
        cleanup: bool,
    ) -> String {
        let tokens = self
            .tokens
            .iter()
            .filter_map(|token| {
                if !cleanup || (token != cls && token != sep && token != pad && token != unk) {
                    Some(token.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut sequence = tokens
            .join(" ")
            .replace(format!(" {}", prefix).as_str(), "");
        if cleanup {
            sequence = sequence
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

        sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::string::Token, normalizer::string::NormalizedString};

    #[test]
    fn test_merge() {
        let encoding = Encoding {
            ids: vec![1],
            type_ids: vec![0],
            tokens: vec![String::from("Hello ")],
            word_indices: vec![Some(0)],
            offsets: vec![Offsets(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            sequence_ranges: None,
            overflowing: None,
        };
        let other = Encoding {
            ids: vec![2],
            type_ids: vec![1],
            tokens: vec![String::from("World!")],
            word_indices: vec![Some(0)],
            offsets: vec![Offsets(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            sequence_ranges: None,
            overflowing: None,
        };
        let merged = encoding.merge_with(other, true);
        let expected = Encoding {
            ids: vec![1, 2],
            type_ids: vec![0, 1],
            tokens: vec![String::from("Hello "), String::from("World!")],
            word_indices: vec![Some(0), Some(0)],
            offsets: vec![Offsets(0, 6), Offsets(6, 12)],
            special_tokens_mask: vec![0, 0],
            attention_mask: vec![1, 1],
            sequence_ranges: Some([(0, 0..1), (1, 1..2)].iter().cloned().collect()),
            overflowing: None,
        };
        assert_eq!(merged, expected);
    }

    #[test]
    fn test_truncate() {
        let encoding = Encoding {
            ids: vec![1, 2, 3],
            type_ids: vec![0, 0, 0],
            tokens: vec!["Hello".into(), "World".into(), "!".into()],
            word_indices: vec![Some(0), Some(1), Some(2)],
            offsets: vec![Offsets(0, 5), Offsets(6, 11), Offsets(11, 12)],
            special_tokens_mask: vec![0, 0, 0],
            attention_mask: vec![1, 1, 1],
            sequence_ranges: None,
            overflowing: None,
        };
        let truncated = encoding.truncate(2, 0);
        let expected = Encoding {
            ids: vec![1, 2],
            type_ids: vec![0, 0],
            tokens: vec!["Hello".into(), "World".into()],
            word_indices: vec![Some(0), Some(1)],
            offsets: vec![Offsets(0, 5), Offsets(6, 11)],
            special_tokens_mask: vec![0, 0],
            attention_mask: vec![1, 1],
            sequence_ranges: None,
            overflowing: Some(vec![Encoding {
                ids: vec![3],
                type_ids: vec![0],
                tokens: vec!["!".into()],
                word_indices: vec![Some(2)],
                offsets: vec![Offsets(11, 12)],
                special_tokens_mask: vec![0],
                attention_mask: vec![1],
                sequence_ranges: None,
                overflowing: None,
            }]),
        };
        assert_eq!(truncated, expected);
    }

    #[test]
    fn test_truncate_all() {
        let encoding = Encoding {
            ids: vec![1, 2, 3],
            type_ids: vec![0, 0, 0],
            tokens: vec!["Hello".into(), "World".into(), "!".into()],
            word_indices: vec![Some(0), Some(1), Some(2)],
            offsets: vec![Offsets(0, 5), Offsets(6, 11), Offsets(11, 12)],
            special_tokens_mask: vec![0, 0, 0],
            attention_mask: vec![1, 1, 1],
            sequence_ranges: None,
            overflowing: None,
        };
        let truncated = encoding.truncate(0, 0);
        let mut expected = Encoding::with_capacity(0);
        expected.overflowing = Some(vec![Encoding {
            ids: vec![1, 2, 3],
            type_ids: vec![0, 0, 0],
            tokens: vec!["Hello".into(), "World".into(), "!".into()],
            word_indices: vec![Some(0), Some(1), Some(2)],
            offsets: vec![Offsets(0, 5), Offsets(6, 11), Offsets(11, 12)],
            special_tokens_mask: vec![0, 0, 0],
            attention_mask: vec![1, 1, 1],
            sequence_ranges: None,
            overflowing: None,
        }]);
        assert_eq!(truncated, expected);
    }

    #[test]
    fn test_decode() {
        let mut encoding = Encoding::<u32>::with_capacity(0);

        encoding.tokens = vec![
            "[CLS]".into(),
            "hello".into(),
            "world".into(),
            "!".into(),
            "[SEP]".into(),
            "[PAD]".into(),
            "[PAD]".into(),
        ];
        assert_eq!(
            encoding.decode("[CLS]", "[SEP]", "[PAD]", "[UNK]", "##", true),
            "hello world!",
        );

        encoding.tokens = vec!["foo".into(), "##bar".into()];
        assert_eq!(encoding.decode("", "", "", "", "##", true), "foobar");

        encoding.tokens = vec!["[UNK]".into()];
        assert_eq!(encoding.decode("", "", "", "[UNK]", "", true), "");
    }

    #[test]
    fn test_into_encoding() {
        let sequence = TokenizedString {
            splits: vec![
                Split {
                    normalized: NormalizedString {
                        original: "a ab abb".to_string(),
                        normalized: "a".to_string(),
                        alignments: vec![Offsets(0, 1)],
                        offset: 0,
                    },
                    tokens: vec![Token {
                        id: 0,
                        value: "a".to_string(),
                        offsets: Offsets(0, 1),
                    }],
                },
                Split {
                    normalized: NormalizedString {
                        original: "a ab abb".to_string(),
                        normalized: "ab".to_string(),
                        alignments: vec![Offsets(2, 4)],
                        offset: 0,
                    },
                    tokens: vec![
                        Token {
                            id: 0,
                            value: "a".to_string(),
                            offsets: Offsets(2, 3),
                        },
                        Token {
                            id: 1,
                            value: "##b".to_string(),
                            offsets: Offsets(3, 4),
                        },
                    ],
                },
                Split {
                    normalized: NormalizedString {
                        original: "a ab abb".to_string(),
                        normalized: "abb".to_string(),
                        alignments: vec![Offsets(5, 8)],
                        offset: 0,
                    },
                    tokens: vec![
                        Token {
                            id: 0,
                            value: "a".to_string(),
                            offsets: Offsets(5, 6),
                        },
                        Token {
                            id: 1,
                            value: "##b".to_string(),
                            offsets: Offsets(6, 7),
                        },
                        Token {
                            id: 1,
                            value: "##b".to_string(),
                            offsets: Offsets(7, 8),
                        },
                    ],
                },
            ],
        };
        let encoding = Encoding::from(sequence);
        assert_eq!(encoding.ids, [0, 0, 1, 0, 1, 1]);
        assert_eq!(encoding.type_ids, [0; 6]);
        assert_eq!(encoding.tokens, ["a", "a", "##b", "a", "##b", "##b"]);
        assert_eq!(
            encoding.word_indices,
            [Some(0), Some(1), Some(1), Some(2), Some(2), Some(2)],
        );
        assert_eq!(
            encoding.offsets,
            [
                Offsets(0, 1),
                Offsets(2, 3),
                Offsets(3, 4),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 8),
            ],
        );
        assert_eq!(encoding.special_tokens_mask, [0; 6]);
        assert_eq!(encoding.attention_mask, [1; 6]);
        assert!(encoding.sequence_ranges.is_none());
        assert!(encoding.overflowing.is_none());
    }

    #[test]
    fn test_into_encoding_empty() {
        let sequence = TokenizedString::<i32> { splits: vec![] };
        let encoding = Encoding::from(sequence);
        assert!(encoding.ids.is_empty());
        assert!(encoding.type_ids.is_empty());
        assert!(encoding.tokens.is_empty());
        assert!(encoding.word_indices.is_empty());
        assert!(encoding.offsets.is_empty());
        assert!(encoding.special_tokens_mask.is_empty());
        assert!(encoding.attention_mask.is_empty());
        assert!(encoding.sequence_ranges.is_none());
        assert!(encoding.overflowing.is_none());
    }
}
