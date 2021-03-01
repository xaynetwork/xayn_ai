use std::{collections::HashMap, iter, ops::Range};

use crate::{model::string::Token, normalizer::string::Offsets};

/// Represents the output of a `Tokenizer`.
#[derive(Clone, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct Encoding {
    /// IDs produced by the `Tokenizer`
    pub(crate) ids: Vec<u32>,
    /// Type of the IDs
    pub(crate) type_ids: Vec<u32>,
    /// Tokens associated to each ID
    pub(crate) tokens: Vec<String>,
    /// Indices of the word associated to each token/ID
    pub(crate) words: Vec<Option<u32>>,
    /// Offsets of the token/ID from the NormalizedString
    pub(crate) offsets: Vec<Offsets>,
    /// Mask identifying special tokens
    pub(crate) special_tokens_mask: Vec<u32>,
    /// Mask identifying padding tokens for the attention mechanism
    pub(crate) attention_mask: Vec<u32>,
    /// Ranges of tokens covered by each sequence. If this is None or empty we consider
    /// there is only one sequence in this Encoding, and that it covers the entire range.
    pub(crate) sequence_ranges: Option<HashMap<usize, Range<usize>>>,
    /// A list of overflowing Encoding generated when we got truncated
    pub(crate) overflowing: Option<Vec<Encoding>>,
}

impl Encoding {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ids: Vec::with_capacity(capacity),
            type_ids: Vec::with_capacity(capacity),
            tokens: Vec::with_capacity(capacity),
            words: Vec::with_capacity(capacity),
            offsets: Vec::with_capacity(capacity),
            special_tokens_mask: Vec::with_capacity(capacity),
            attention_mask: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Returns the total length of this Encoding.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether this Encoding is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Returns the number of sequences combined in this Encoding.
    pub fn n_sequences(&self) -> usize {
        self.sequence_ranges
            .as_ref()
            .map(|sequence_ranges| sequence_ranges.len())
            .unwrap_or(1)
    }

    /// Sets the given sequence id for the whole range of tokens contained in this Encoding.
    pub fn with_sequence_id(mut self, id: usize) -> Self {
        self.sequence_ranges = Some(iter::once((id, 0..self.len())).collect());
        self
    }

    pub fn get_sequence_ids(&self) -> Vec<Option<usize>> {
        let mut sequences = vec![None; self.len()];
        for seq_id in 0..self.n_sequences() {
            let range = self.sequence_range(seq_id);
            let seq_len = range.len();
            sequences.splice(range, std::iter::repeat(Some(seq_id)).take(seq_len));
        }
        sequences
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    pub fn tokens(&self) -> &[String] {
        &self.tokens[..]
    }

    pub fn words(&self) -> &[Option<u32>] {
        &self.words
    }

    pub fn offsets(&self) -> &[Offsets] {
        &self.offsets
    }

    pub fn special_tokens_mask(&self) -> &[u32] {
        &self.special_tokens_mask
    }

    pub fn attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub fn overflowing(&self) -> Option<&[Encoding]> {
        self.overflowing
            .as_ref()
            .map(|overflowing| overflowing.as_slice())
    }

    fn process_tokens_with_offsets_mut(&mut self, f: impl FnMut((usize, (&String, &mut Offsets)))) {
        self.tokens
            .iter()
            .zip(self.offsets.iter_mut())
            .enumerate()
            .for_each(f)
    }

    /// Returns the range to target to retrieve something (word_id, offsets, ..) related to the
    /// given sequence id
    fn sequence_range(&self, sequence_id: usize) -> Range<usize> {
        self.sequence_ranges
            .as_ref()
            .map(|sequence_ranges| sequence_ranges.get(&sequence_id).cloned())
            .flatten()
            .unwrap_or_else(|| 0..self.len())
    }

    /// Returns the index of the sequence containing the given token
    fn token_to_sequence(&self, token: usize) -> Option<usize> {
        if token > self.len() {
            None
        } else if self.sequence_ranges.is_none() {
            Some(0)
        } else {
            self.sequence_ranges
                .as_ref()
                .map(|sequence_ranges| {
                    sequence_ranges.iter().find_map(|(seq_id, range)| {
                        if range.contains(&token) {
                            Some(*seq_id)
                        } else {
                            None
                        }
                    })
                })
                .flatten()
        }
    }

    /// Get the encoded tokens corresponding to the word at the given index in the input sequence,
    /// with the form (start_token, end_token + 1)
    fn word_to_tokens(&self, word: u32, sequence_id: usize) -> Option<Offsets> {
        let mut start = None;
        let mut end = None;
        let sequence_range = self.sequence_range(sequence_id);

        self.words
            .get(sequence_range.clone())?
            .iter()
            .enumerate()
            .take_while(|(_, w)| **w <= Some(word))
            .filter(|(_, w)| **w == Some(word))
            .for_each(|(i, _)| {
                if start.is_none() || Some(i) < start {
                    start = Some(i);
                }
                if end.is_none() || Some(i) >= end {
                    end = Some(i + 1);
                }
            });

        if let (Some(start), Some(end)) = (start, end) {
            Some(Offsets(
                sequence_range.start + start,
                sequence_range.start + end,
            ))
        } else {
            None
        }
    }

    /// Get the offsets of the word at the given index in the input sequence.
    fn word_to_chars(&self, word: u32, sequence_id: usize) -> Option<Offsets> {
        self.word_to_tokens(word, sequence_id)
            .and_then(|Offsets(start, end)| {
                if end == 0 {
                    None
                } else {
                    Some(Offsets(self.offsets[start].0, self.offsets[end - 1].1))
                }
            })
    }

    /// Get the offsets of the token at the given index.
    fn token_to_chars(&self, token: usize) -> Option<(usize, Offsets)> {
        Some((
            self.token_to_sequence(token)?,
            self.offsets.get(token).copied()?,
        ))
    }

    /// Get the word that contains the token at the given index.
    fn token_to_word(&self, token: usize) -> Option<(usize, u32)> {
        Some((
            self.token_to_sequence(token)?,
            self.words.get(token).copied().flatten()?,
        ))
    }

    /// Get the token that contains the given char.
    fn char_to_token(&self, pos: usize, sequence_id: usize) -> Option<usize> {
        let sequence_range = self.sequence_range(sequence_id);
        let sequence_start = sequence_range.start;
        self.offsets
            .get(sequence_range)?
            .iter()
            .position(|Offsets(start, end)| pos >= *start && pos < *end)
            .map(|pos| sequence_start + pos)
    }

    /// Get the word that contains the given char.
    fn char_to_word(&self, pos: usize, sequence_id: usize) -> Option<u32> {
        Some(
            self.char_to_token(pos, sequence_id)
                .and_then(|token| self.token_to_word(token))?
                .1,
        )
    }

    /// Extends the encoding from the "slice" `&from[skip..skip+take]`.
    #[inline]
    fn extend(&mut self, from: &Self, skip: usize, take: usize) {
        debug_assert!(self.overflowing.is_none());
        debug_assert!(self.sequence_ranges.is_none());

        self.ids.extend(from.ids.iter().skip(skip).take(take));
        self.type_ids
            .extend(from.type_ids.iter().skip(skip).take(take));
        self.tokens
            .extend(from.tokens.iter().skip(skip).take(take).cloned());
        self.words.extend(from.words.iter().skip(skip).take(take));
        self.offsets
            .extend(from.offsets.iter().skip(skip).take(take));
        self.special_tokens_mask
            .extend(from.special_tokens_mask.iter().skip(skip).take(take));
        self.attention_mask
            .extend(from.attention_mask.iter().skip(skip).take(take));
    }

    /// Drains the encoding and chains the drainage with the "slice" `&from[skip..skip+take]`.
    #[inline]
    fn drain_chain(&mut self, from: &Self, skip: usize, take: usize) -> Self {
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
            words: self
                .words
                .drain(..)
                .chain(from.words.iter().skip(skip).take(take).copied())
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
            ..Self::default()
        }
    }

    /// Truncate the current `Encoding`.
    ///
    /// # Panics
    /// Panics if `stride >= max_len`.
    pub(crate) fn truncate(mut self, max_len: usize, stride: usize) -> Self {
        if max_len >= self.len() {
            return self;
        }

        if max_len == 0 {
            let mut encoding = Self::with_capacity(0);
            encoding.overflowing = Some(vec![self]);
            return encoding;
        }

        // Get the main overflowing part
        let overflowings = &Self {
            ids: self.ids.split_off(max_len),
            type_ids: self.type_ids.split_off(max_len),
            tokens: self.tokens.split_off(max_len),
            words: self.words.split_off(max_len),
            offsets: self.offsets.split_off(max_len),
            special_tokens_mask: self.special_tokens_mask.split_off(max_len),
            attention_mask: self.attention_mask.split_off(max_len),
            ..Self::default()
        };

        // When truncating, we loose the `sequence_ranges` information.
        self.sequence_ranges = None;

        // Now we need to separate the overflowing part into as many Encoding as needed
        assert!(stride < max_len);
        let part_size = max_len - stride;
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

    pub(crate) fn pad(
        mut self,
        target_length: usize,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
    ) -> Self {
        // Dispatch call to all the overflowings first
        self.overflowing = self.overflowing.map(|overflowing| {
            overflowing
                .into_iter()
                .map(|encoding| encoding.pad(target_length, pad_id, pad_type_id, pad_token))
                .collect()
        });

        // Then check if we should pad ourself
        if self.len() >= target_length {
            // We just do nothing if the wanted padding length is smaller than us
            return self;
        }
        let pad_length = target_length - self.len();

        self.ids.extend(iter::repeat(pad_id).take(pad_length));
        self.type_ids
            .extend(iter::repeat(pad_type_id).take(pad_length));
        self.tokens
            .extend(iter::repeat(pad_token.to_string()).take(pad_length));
        self.words.extend(iter::repeat(None).take(pad_length));
        self.attention_mask.extend(iter::repeat(0).take(pad_length));
        self.special_tokens_mask
            .extend(iter::repeat(1).take(pad_length));
        self.offsets
            .extend(iter::repeat(Offsets(0, 0)).take(pad_length));

        self
    }

    /// Merge all Encodings together
    pub fn merge(encodings: impl IntoIterator<Item = Encoding>, growing_offsets: bool) -> Self {
        encodings
            .into_iter()
            .fold(Encoding::default(), |encoding, other| {
                encoding.merge_with(other, growing_offsets)
            })
    }

    /// Merge ourself with the given `Encoding`. Happens in place.
    pub fn merge_with(mut self, other: Encoding, growing_offsets: bool) -> Self {
        // Handle merging the overflowing parts too: Combine them all
        // In most of the cases, we expect `other.overflowing.len() == 0`
        let mut overflowings = vec![];

        // 1. All our overflowings with all the others
        self.overflowing.as_ref().map(|overflowing| {
            for self_o in overflowing {
                // 1. The other itself
                let n_encoding = self_o.clone();
                let n_encoding = n_encoding.merge_with(other.clone(), growing_offsets);
                overflowings.push(n_encoding);

                // 2. Its overflowings (this should rarely happen...)
                other.overflowing.as_ref().map(|overflowing| {
                    for other_o in overflowing {
                        let n_encoding = self_o.clone();
                        let n_encoding = n_encoding.merge_with(other_o.clone(), growing_offsets);
                        overflowings.push(n_encoding);
                    }
                });
            }
        });

        // 2. Ourself with all the other overflowings (this should rarely happen too...)
        other.overflowing.as_ref().map(|overflowing| {
            for other_o in overflowing {
                let n_encoding = self.clone();
                let n_encoding = n_encoding.merge_with(other_o.clone(), growing_offsets);
                overflowings.push(n_encoding);
            }
        });

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
        self.words.extend(other.words);

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

    pub fn decode(&self, unk: impl AsRef<str>, prefix: impl AsRef<str>, cleanup: bool) -> String {
        let tokens = self
            .tokens
            .iter()
            .filter_map(|token| {
                if !cleanup || token != unk.as_ref() {
                    Some(token.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut string = tokens
            .join(" ")
            .replace(format!(" {}", prefix.as_ref()).as_str(), "");
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

impl std::iter::FromIterator<Encoding> for Encoding {
    fn from_iter<I: IntoIterator<Item = Encoding>>(iter: I) -> Self {
        Self::merge(iter, false)
    }
}

impl std::iter::FromIterator<(Token, Option<u32>)> for Encoding {
    fn from_iter<I: IntoIterator<Item = (Token, Option<u32>)>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let len = iter.by_ref().count();
        let ids = iter.by_ref().map(|(token, _)| token.id).collect::<Vec<_>>();
        let words = iter.by_ref().map(|(_, word)| word).collect();
        let offsets = iter.by_ref().map(|(token, _)| token.offsets).collect();
        let tokens = iter.map(|(token, _)| token.value).collect();

        Self {
            ids,
            type_ids: vec![0; len],
            tokens,
            words,
            offsets,
            special_tokens_mask: vec![0; len],
            attention_mask: vec![1; len],
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        let encoding = Encoding {
            ids: vec![1],
            type_ids: vec![0],
            tokens: vec![String::from("Hello ")],
            words: vec![Some(0)],
            offsets: vec![Offsets(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            ..Default::default()
        };
        let other = Encoding {
            ids: vec![2],
            type_ids: vec![1],
            tokens: vec![String::from("World!")],
            words: vec![Some(0)],
            offsets: vec![Offsets(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            ..Default::default()
        };
        let merged = encoding.merge_with(other, true);
        let expected = Encoding {
            ids: vec![1, 2],
            type_ids: vec![0, 1],
            tokens: vec![String::from("Hello "), String::from("World!")],
            words: vec![Some(0), Some(0)],
            offsets: vec![Offsets(0, 6), Offsets(6, 12)],
            special_tokens_mask: vec![0, 0],
            attention_mask: vec![1, 1],
            sequence_ranges: Some([(0, 0..1), (1, 1..2)].iter().cloned().collect()),
            ..Encoding::default()
        };
        assert_eq!(merged, expected);
    }

    #[test]
    fn test_truncate() {
        let encoding = Encoding {
            ids: vec![1, 2, 3],
            type_ids: vec![0, 0, 0],
            tokens: vec!["Hello".into(), "World".into(), "!".into()],
            words: vec![Some(0), Some(1), Some(2)],
            offsets: vec![Offsets(0, 5), Offsets(6, 11), Offsets(11, 12)],
            special_tokens_mask: vec![0, 0, 0],
            attention_mask: vec![1, 1, 1],
            ..Encoding::default()
        };
        let truncated = encoding.truncate(2, 0);
        let expected = Encoding {
            ids: vec![1, 2],
            type_ids: vec![0, 0],
            tokens: vec!["Hello".into(), "World".into()],
            words: vec![Some(0), Some(1)],
            offsets: vec![Offsets(0, 5), Offsets(6, 11)],
            special_tokens_mask: vec![0, 0],
            attention_mask: vec![1, 1],
            overflowing: Some(vec![Encoding {
                ids: vec![3],
                type_ids: vec![0],
                tokens: vec!["!".into()],
                words: vec![Some(2)],
                offsets: vec![Offsets(11, 12)],
                special_tokens_mask: vec![0],
                attention_mask: vec![1],
                ..Encoding::default()
            }]),
            ..Encoding::default()
        };
        assert_eq!(truncated, expected);
    }

    #[test]
    fn test_truncate_all() {
        let encoding = Encoding {
            ids: vec![1, 2, 3],
            type_ids: vec![0, 0, 0],
            tokens: vec!["Hello".into(), "World".into(), "!".into()],
            words: vec![Some(0), Some(1), Some(2)],
            offsets: vec![Offsets(0, 5), Offsets(6, 11), Offsets(11, 12)],
            special_tokens_mask: vec![0, 0, 0],
            attention_mask: vec![1, 1, 1],
            ..Encoding::default()
        };
        let truncated = encoding.truncate(0, 0);
        let expected = Encoding {
            overflowing: Some(vec![Encoding {
                ids: vec![1, 2, 3],
                type_ids: vec![0, 0, 0],
                tokens: vec!["Hello".into(), "World".into(), "!".into()],
                words: vec![Some(0), Some(1), Some(2)],
                offsets: vec![Offsets(0, 5), Offsets(6, 11), Offsets(11, 12)],
                special_tokens_mask: vec![0, 0, 0],
                attention_mask: vec![1, 1, 1],
                ..Encoding::default()
            }]),
            ..Encoding::default()
        };
        assert_eq!(truncated, expected);
    }

    #[test]
    fn test_mappings() {
        let encoding = Encoding {
            ids: (0..11).collect(),
            tokens: vec![
                // First sequence:
                "He".into(),
                "llo".into(),
                "won".into(),
                "der".into(),
                "ful".into(),
                "friend".into(),
                "!".into(),
                // Second sequence:
                "How".into(),
                "are".into(),
                "you".into(),
                "?".into(),
            ],
            offsets: vec![
                // First sequence:
                Offsets(0, 2),
                Offsets(2, 5),
                Offsets(7, 10),
                Offsets(10, 13),
                Offsets(13, 16),
                Offsets(17, 23),
                Offsets(23, 24),
                // Second sequence:
                Offsets(0, 3),
                Offsets(4, 7),
                Offsets(8, 11),
                Offsets(11, 12),
            ],
            words: vec![
                // First sequence:
                Some(0),
                Some(0),
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                Some(3),
                // Second sequence:
                Some(0),
                Some(1),
                Some(2),
                Some(3),
            ],
            sequence_ranges: Some([(0, 0..7), (1, 7..11)].iter().cloned().collect()),
            ..Encoding::default()
        };

        assert_eq!(encoding.word_to_tokens(0, 0), Some(Offsets(0, 2)));
        assert_eq!(encoding.word_to_tokens(1, 0), Some(Offsets(2, 5)));
        assert_eq!(encoding.word_to_tokens(2, 0), Some(Offsets(5, 6)));
        assert_eq!(encoding.word_to_tokens(3, 0), Some(Offsets(6, 7)));
        assert_eq!(encoding.word_to_tokens(0, 1), Some(Offsets(7, 8)));
        assert_eq!(encoding.word_to_tokens(1, 1), Some(Offsets(8, 9)));
        assert_eq!(encoding.word_to_tokens(2, 1), Some(Offsets(9, 10)));
        assert_eq!(encoding.word_to_tokens(3, 1), Some(Offsets(10, 11)));

        assert_eq!(encoding.word_to_chars(0, 0), Some(Offsets(0, 5)));
        assert_eq!(encoding.word_to_chars(1, 0), Some(Offsets(7, 16)));
        assert_eq!(encoding.word_to_chars(0, 1), Some(Offsets(0, 3)));
        assert_eq!(encoding.word_to_chars(1, 1), Some(Offsets(4, 7)));

        assert_eq!(encoding.token_to_chars(0), Some((0, Offsets(0, 2))));
        assert_eq!(encoding.token_to_chars(1), Some((0, Offsets(2, 5))));
        assert_eq!(encoding.token_to_chars(7), Some((1, Offsets(0, 3))));
        assert_eq!(encoding.token_to_chars(9), Some((1, Offsets(8, 11))));

        assert_eq!(encoding.token_to_word(1), Some((0, 0)));
        assert_eq!(encoding.token_to_word(2), Some((0, 1)));
        assert_eq!(encoding.token_to_word(7), Some((1, 0)));
        assert_eq!(encoding.token_to_word(9), Some((1, 2)));
        assert_eq!(encoding.token_to_word(11), None);

        assert_eq!(encoding.char_to_token(3, 0), Some(1));
        assert_eq!(encoding.char_to_token(8, 0), Some(2));
        assert_eq!(encoding.char_to_token(16, 0), None);
        assert_eq!(encoding.char_to_token(23, 0), Some(6));
        assert_eq!(encoding.char_to_token(2, 1), Some(7));
        assert_eq!(encoding.char_to_token(9, 1), Some(9));

        assert_eq!(encoding.char_to_word(3, 0), Some(0));
        assert_eq!(encoding.char_to_word(8, 0), Some(1));
        assert_eq!(encoding.char_to_word(16, 0), None);
        assert_eq!(encoding.char_to_word(23, 0), Some(3));
        assert_eq!(encoding.char_to_word(2, 1), Some(0));
        assert_eq!(encoding.char_to_word(9, 1), Some(2));
    }
}
