use std::{collections::HashMap, iter, ops::Range};

use crate::{normalizer::pattern::Offsets, pipeline::Token};

/// Represents the output of a `Tokenizer`.
#[derive(Default, PartialEq, Debug, Clone)]
pub struct Encoding {
    /// IDs produced by the `Tokenizer`
    pub(crate) ids: Vec<u32>,
    /// Type of the IDs
    pub(crate) type_ids: Vec<u32>,
    /// Tokens associated to each ID
    pub(crate) tokens: Vec<String>,
    /// Indice of the word associated to each token/ID
    pub(crate) words: Vec<Option<u32>>,
    /// Offsets of the token/ID from the NormalizedString
    pub(crate) offsets: Vec<Offsets>,
    /// Mask identifying special tokens
    pub(crate) special_tokens_mask: Vec<u32>,
    /// Mask identifying padding tokens for the attention mechanism
    pub(crate) attention_mask: Vec<u32>,
    /// A list of overflowing Encoding generated when we got truncated
    pub(crate) overflowing: Vec<Encoding>,
    /// Ranges of tokens covered by each sequence. If this is empty we consider
    /// there is only one sequence in this Encoding, and that it covers the entire range.
    pub(crate) sequence_ranges: HashMap<usize, Range<usize>>,
}
impl Encoding {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ids: Vec<u32>,
        type_ids: Vec<u32>,
        tokens: Vec<String>,
        words: Vec<Option<u32>>,
        offsets: Vec<Offsets>,
        special_tokens_mask: Vec<u32>,
        attention_mask: Vec<u32>,
        overflowing: Vec<Encoding>,
        sequence_ranges: HashMap<usize, Range<usize>>,
    ) -> Self {
        Encoding {
            ids,
            type_ids,
            tokens,
            words,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
            sequence_ranges,
        }
    }

    pub fn with_capacity(len: usize) -> Self {
        Encoding {
            ids: Vec::with_capacity(len),
            type_ids: Vec::with_capacity(len),
            tokens: Vec::with_capacity(len),
            words: Vec::with_capacity(len),
            offsets: Vec::with_capacity(len),
            special_tokens_mask: Vec::with_capacity(len),
            attention_mask: Vec::with_capacity(len),
            overflowing: vec![],
            sequence_ranges: HashMap::new(),
        }
    }

    pub fn from_tokens(tokens: Vec<Token>, type_id: u32) -> Self {
        let length = tokens.len();
        let (ids, tokens, offsets) = tokens.into_iter().fold(
            (
                Vec::with_capacity(length),
                Vec::with_capacity(length),
                Vec::with_capacity(length),
            ),
            |(mut ids, mut tokens, mut offsets), t| {
                ids.push(t.id);
                tokens.push(t.value);
                offsets.push(t.offsets);
                (ids, tokens, offsets)
            },
        );

        Encoding {
            ids,
            tokens,
            offsets,
            words: vec![None; length],
            type_ids: vec![type_id; length],
            attention_mask: vec![1; length],
            special_tokens_mask: vec![0; length],
            overflowing: vec![],
            sequence_ranges: HashMap::new(),
        }
    }

    /// Whether this Encoding is empty
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Return the total length of this Encoding
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Return the number of sequences combined in this Encoding
    pub fn n_sequences(&self) -> usize {
        if self.sequence_ranges.is_empty() {
            1
        } else {
            self.sequence_ranges.len()
        }
    }

    /// Set the given sequence id for the whole range of tokens contained in this Encoding
    pub fn set_sequence_id(&mut self, sequence_id: usize) {
        self.sequence_ranges.insert(sequence_id, 0..self.len());
    }

    pub fn get_tokens(&self) -> &[String] {
        &self.tokens[..]
    }

    pub fn get_word_ids(&self) -> &[Option<u32>] {
        &self.words
    }

    pub fn get_word_ids_mut(&mut self) -> &mut [Option<u32>] {
        &mut self.words
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

    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn get_type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    pub fn get_offsets(&self) -> &[Offsets] {
        &self.offsets
    }

    pub fn get_offsets_mut(&mut self) -> &mut [Offsets] {
        &mut self.offsets
    }

    pub fn get_special_tokens_mask(&self) -> &[u32] {
        &self.special_tokens_mask
    }

    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub fn get_overflowing(&self) -> &Vec<Encoding> {
        &self.overflowing
    }

    pub fn get_overflowing_mut(&mut self) -> &mut Vec<Encoding> {
        &mut self.overflowing
    }

    pub fn take_overflowing(&mut self) -> Vec<Encoding> {
        std::mem::replace(&mut self.overflowing, vec![])
    }

    pub(crate) fn process_tokens_with_offsets_mut<F>(&mut self, func: F)
    where
        F: FnMut((usize, (&String, &mut Offsets))),
    {
        self.tokens
            .iter()
            .zip(self.offsets.iter_mut())
            .enumerate()
            .for_each(func)
    }

    /// Returns the range to target to retrieve something (word_id, offsets, ..) related to the
    /// given sequence id
    fn sequence_range(&self, sequence_id: usize) -> Range<usize> {
        self.sequence_ranges
            .get(&sequence_id)
            .cloned()
            .unwrap_or(0..self.len())
    }

    /// Returns the index of the sequence containing the given token
    pub fn token_to_sequence(&self, token: usize) -> Option<usize> {
        if token > self.len() {
            None
        } else if self.sequence_ranges.is_empty() {
            Some(0)
        } else {
            self.sequence_ranges.iter().find_map(|(seq_id, range)| {
                if range.contains(&token) {
                    Some(*seq_id)
                } else {
                    None
                }
            })
        }
    }

    /// Get the encoded tokens corresponding to the word at the given index in the input sequence,
    /// with the form (start_token, end_token + 1)
    pub fn word_to_tokens(&self, word: u32, sequence_id: usize) -> Option<(usize, usize)> {
        let (mut start, mut end) = (None, None);
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
            Some((sequence_range.start + start, sequence_range.start + end))
        } else {
            None
        }
    }

    /// Get the offsets of the word at the given index in the input sequence.
    pub fn word_to_chars(&self, word: u32, sequence_id: usize) -> Option<Offsets> {
        self.word_to_tokens(word, sequence_id)
            .and_then(|(start, end)| {
                if end == 0 {
                    None
                } else {
                    Some((self.offsets[start].0, self.offsets[end - 1].1))
                }
            })
    }

    /// Get the offsets of the token at the given index.
    pub fn token_to_chars(&self, token: usize) -> Option<(usize, Offsets)> {
        Some((
            self.token_to_sequence(token)?,
            self.offsets.get(token).copied()?,
        ))
    }

    /// Get the word that contains the token at the given index.
    pub fn token_to_word(&self, token: usize) -> Option<(usize, u32)> {
        Some((
            self.token_to_sequence(token)?,
            self.words.get(token).copied().flatten()?,
        ))
    }

    /// Get the token that contains the given char.
    pub fn char_to_token(&self, pos: usize, sequence_id: usize) -> Option<usize> {
        let sequence_range = self.sequence_range(sequence_id);

        self.offsets
            .get(sequence_range.clone())?
            .iter()
            .position(|(start, end)| pos >= *start && pos < *end)
            .map(|pos| sequence_range.start + pos)
    }

    /// Get the word that contains the given char.
    pub fn char_to_word(&self, pos: usize, sequence_id: usize) -> Option<u32> {
        Some(
            self.char_to_token(pos, sequence_id)
                .and_then(|token| self.token_to_word(token))?
                .1,
        )
    }

    /// Extends the encoding from the "slice" `&from[skip..skip+take]`.
    #[inline]
    fn extend(&mut self, from: &Self, skip: usize, take: usize) {
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
    /// Panic if `stride >= max_len`
    pub fn truncate(mut self, max_len: usize, stride: usize) -> Self {
        if max_len >= self.ids.len() {
            return self;
        }

        if max_len == 0 {
            let overflowing = std::mem::replace(&mut self, Self::with_capacity(0));
            self.overflowing.push(overflowing);
            return self;
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
        self.sequence_ranges.clear();

        // Now we need to separate the overflowing part into as many Encoding as needed
        assert!(stride < max_len);
        let part_size = max_len - stride;
        self.overflowing = (0..overflowings.ids.len())
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
            .collect();

        self
    }

    /// Merge all Encodings together
    pub fn merge<I: IntoIterator<Item = Encoding>>(encodings: I, growing_offsets: bool) -> Self {
        let mut encoding = Encoding::default();

        for sub in encodings {
            encoding.merge_with(sub, growing_offsets);
        }

        encoding
    }

    /// Merge ourself with the given `Encoding`. Happens in place.
    pub fn merge_with(&mut self, pair: Encoding, growing_offsets: bool) {
        // Handle merging the overflowing parts too: Combine them all
        // In most of the cases, we expect `pair.overflowing.len() == 0`
        let mut overflowings = vec![];

        // 1. All our overflowings with all the others
        for self_o in &self.overflowing {
            // 1. The pair itself
            let mut n_encoding = self_o.clone();
            n_encoding.merge_with(pair.clone(), growing_offsets);
            overflowings.push(n_encoding);

            // 2. Its overflowings (this should rarely happen...)
            for other_o in &pair.overflowing {
                let mut n_encoding = self_o.clone();
                n_encoding.merge_with(other_o.clone(), growing_offsets);
                overflowings.push(n_encoding);
            }
        }
        // 2. Ourself with all the other overflowings (this should rarely happen too...)
        for other_o in &pair.overflowing {
            let mut n_encoding = self.clone();
            n_encoding.merge_with(other_o.clone(), growing_offsets);
            overflowings.push(n_encoding);
        }

        // Finish by merging ourself with the other encoding
        let original_self_len = self.len(); // Must be before any modification to self.ids

        self.sequence_ranges
            .extend(pair.sequence_ranges.into_iter().map(|(seq_id, range)| {
                (
                    seq_id,
                    original_self_len + range.start..original_self_len + range.end,
                )
            }));
        self.ids.extend(pair.ids);
        self.type_ids.extend(pair.type_ids);
        self.tokens.extend(pair.tokens);
        self.words.extend(pair.words);

        let starting_offset = if growing_offsets {
            self.offsets.last().map_or(0, |o| o.1)
        } else {
            0
        };
        self.offsets.extend(
            pair.offsets
                .into_iter()
                .map(|(start, end)| (start + starting_offset, end + starting_offset))
                .collect::<Vec<_>>(),
        );
        self.special_tokens_mask.extend(pair.special_tokens_mask);
        self.attention_mask.extend(pair.attention_mask);
        self.overflowing = overflowings;
    }

    pub fn pad(
        mut self,
        target_length: usize,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
    ) -> Self {
        // Dispatch call to all the overflowings first
        self.overflowing = self
            .overflowing
            .into_iter()
            .map(|encoding| encoding.pad(target_length, pad_id, pad_type_id, pad_token))
            .collect();

        // Then check if we should pad ourself
        if self.ids.len() >= target_length {
            // We just do nothing if the wanted padding length is smaller than us
            return self;
        }
        let pad_length = target_length - self.ids.len();

        self.ids.extend(iter::repeat(pad_id).take(pad_length));
        self.type_ids
            .extend(iter::repeat(pad_type_id).take(pad_length));
        self.tokens
            .extend(iter::repeat(pad_token.to_string()).take(pad_length));
        self.words.extend(iter::repeat(None).take(pad_length));
        self.attention_mask.extend(iter::repeat(0).take(pad_length));
        self.special_tokens_mask
            .extend(iter::repeat(1).take(pad_length));
        self.offsets.extend(iter::repeat((0, 0)).take(pad_length));

        self
    }
}

impl std::iter::FromIterator<Encoding> for Encoding {
    fn from_iter<I: IntoIterator<Item = Encoding>>(iter: I) -> Self {
        Self::merge(iter, false)
    }
}

impl std::iter::FromIterator<(u32, String, (usize, usize), Option<u32>, u32)> for Encoding {
    fn from_iter<I: IntoIterator<Item = (u32, String, (usize, usize), Option<u32>, u32)>>(
        iter: I,
    ) -> Self {
        let items = iter.into_iter();
        let (lower, upper) = items.size_hint();
        let length = upper.unwrap_or(lower);
        let mut encoding = Self::with_capacity(length);

        for (id, token, offsets, word, type_id) in items {
            encoding.ids.push(id);
            encoding.tokens.push(token);
            encoding.offsets.push(offsets);
            encoding.type_ids.push(type_id);
            encoding.words.push(word);
            encoding.special_tokens_mask.push(0);
            encoding.attention_mask.push(1);
        }

        encoding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn merge_encodings() {
        let mut a = Encoding {
            ids: vec![1],
            type_ids: vec![0],
            tokens: vec![String::from("Hello ")],
            words: vec![Some(0)],
            offsets: vec![(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            ..Default::default()
        };
        let b = Encoding {
            ids: vec![2],
            type_ids: vec![1],
            tokens: vec![String::from("World!")],
            words: vec![Some(0)],
            offsets: vec![(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            ..Default::default()
        };
        a.merge_with(b, true);

        assert_eq!(
            a,
            Encoding {
                ids: vec![1, 2],
                type_ids: vec![0, 1],
                tokens: vec![String::from("Hello "), String::from("World!")],
                words: vec![Some(0), Some(0)],
                offsets: vec![(0, 6), (6, 12)],
                special_tokens_mask: vec![0, 0],
                attention_mask: vec![1, 1],
                ..Default::default()
            }
        );
    }

    #[test]
    fn truncate() {
        let a = Encoding {
            ids: vec![1, 2, 3],
            type_ids: vec![0, 0, 0],
            tokens: vec![
                String::from("Hello"),
                String::from("World"),
                String::from("!"),
            ],
            words: vec![Some(0), Some(1), Some(2)],
            offsets: vec![(0, 5), (6, 11), (11, 12)],
            special_tokens_mask: vec![0, 0, 0],
            attention_mask: vec![1, 1, 1],
            ..Default::default()
        };
        let a = a.truncate(2, 0);

        assert_eq!(
            a,
            Encoding {
                ids: vec![1, 2],
                type_ids: vec![0, 0],
                tokens: vec![String::from("Hello"), String::from("World")],
                words: vec![Some(0), Some(1)],
                offsets: vec![(0, 5), (6, 11)],
                special_tokens_mask: vec![0, 0],
                attention_mask: vec![1, 1],
                overflowing: vec![Encoding {
                    ids: vec![3],
                    type_ids: vec![0],
                    tokens: vec![String::from("!")],
                    words: vec![Some(2)],
                    offsets: vec![(11, 12)],
                    special_tokens_mask: vec![0],
                    attention_mask: vec![1],
                    ..Default::default()
                }],
                ..Default::default()
            }
        );
    }

    #[test]
    fn truncate_to_empty() {
        let a = Encoding {
            ids: vec![1, 2, 3],
            type_ids: vec![0, 0, 0],
            tokens: vec![
                String::from("Hello"),
                String::from("World"),
                String::from("!"),
            ],
            words: vec![Some(0), Some(1), Some(2)],
            offsets: vec![(0, 5), (6, 11), (11, 12)],
            special_tokens_mask: vec![0, 0, 0],
            attention_mask: vec![1, 1, 1],
            ..Default::default()
        };
        let a = a.truncate(0, 0);

        assert_eq!(
            a,
            Encoding {
                overflowing: vec![Encoding {
                    ids: vec![1, 2, 3],
                    type_ids: vec![0, 0, 0],
                    tokens: vec![
                        String::from("Hello"),
                        String::from("World"),
                        String::from("!"),
                    ],
                    words: vec![Some(0), Some(1), Some(2)],
                    offsets: vec![(0, 5), (6, 11), (11, 12)],
                    special_tokens_mask: vec![0, 0, 0],
                    attention_mask: vec![1, 1, 1],
                    overflowing: vec![],
                    ..Default::default()
                }],
                ..Default::default()
            }
        );
    }

    #[test]
    fn mappings() {
        let encoding = Encoding {
            ids: vec![0; 11], // Needed for Encoding::len
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
                (0, 2),
                (2, 5),
                (7, 10),
                (10, 13),
                (13, 16),
                (17, 23),
                (23, 24),
                // Second sequence:
                (0, 3),
                (4, 7),
                (8, 11),
                (11, 12),
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
            sequence_ranges: HashMap::from_iter(vec![(0, 0..7), (1, 7..11)]),
            ..Default::default()
        };
        assert_eq!(encoding.word_to_tokens(0, 0), Some((0, 2)));
        assert_eq!(encoding.word_to_tokens(1, 0), Some((2, 5)));
        assert_eq!(encoding.word_to_tokens(2, 0), Some((5, 6)));
        assert_eq!(encoding.word_to_tokens(3, 0), Some((6, 7)));
        assert_eq!(encoding.word_to_tokens(0, 1), Some((7, 8)));
        assert_eq!(encoding.word_to_tokens(1, 1), Some((8, 9)));
        assert_eq!(encoding.word_to_tokens(2, 1), Some((9, 10)));
        assert_eq!(encoding.word_to_tokens(3, 1), Some((10, 11)));

        assert_eq!(encoding.word_to_chars(0, 0), Some((0, 5)));
        assert_eq!(encoding.word_to_chars(1, 0), Some((7, 16)));
        assert_eq!(encoding.word_to_chars(0, 1), Some((0, 3)));
        assert_eq!(encoding.word_to_chars(1, 1), Some((4, 7)));

        assert_eq!(encoding.token_to_chars(0), Some((0, (0, 2))));
        assert_eq!(encoding.token_to_chars(1), Some((0, (2, 5))));
        assert_eq!(encoding.token_to_chars(7), Some((1, (0, 3))));
        assert_eq!(encoding.token_to_chars(9), Some((1, (8, 11))));

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
