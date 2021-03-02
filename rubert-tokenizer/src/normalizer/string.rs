use std::ops::{Bound, RangeBounds};

use unicode_normalization_alignments::UnicodeNormalization;

use crate::{normalizer::pattern::Pattern, Error};

/// Offsets of a subsequence relative to a sequence.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(test, derive(Debug))]
pub struct Offsets(pub usize, pub usize);

impl Offsets {
    /// Returns the range covered by a slice of alignments.
    fn expand_alignments(alignments: &[Offsets]) -> Option<std::ops::Range<usize>> {
        if alignments.is_empty() {
            None
        } else {
            let start = alignments[0].0;
            let end = alignments[alignments.len() - 1].1;
            Some(start..end)
        }
    }
}

/// A range for sequences, either relative to the original or the normalized sequence.
#[derive(Clone)]
pub enum Range<R> {
    Original(R),
    Normalized(R),
}

impl<R> Range<R>
where
    R: RangeBounds<usize>,
{
    fn inner(&self) -> &R {
        match self {
            Range::Original(range) => range,
            Range::Normalized(range) => range,
        }
    }

    /// Converts to a standard range.
    ///
    /// Requires the length of the represented sequences in case the inner range is unbounded.
    fn into_range(self, len: usize) -> std::ops::Range<usize> {
        let range = self.inner();
        let start = match range.start_bound() {
            Bound::Included(idx) => *idx,
            Bound::Excluded(idx) => *idx + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(idx) => *idx + 1,
            Bound::Excluded(idx) => *idx,
            Bound::Unbounded => len,
        };

        start..end
    }

    /// Converts from one referential to the other one wrt a sequence.
    ///
    /// Returns `None` when targeting something that is out of range.
    pub fn convert(self, sequence: &NormalizedString) -> Option<std::ops::Range<usize>> {
        let len_original = sequence.original.len();
        let len_normalized = sequence.normalized.len();

        let (target, original) = match self {
            Range::Original(_) => (self.into_range(len_original), true),
            Range::Normalized(_) => (self.into_range(len_normalized), false),
        };

        // If we target an empty range, let's return the same
        if target.start == target.end {
            return Some(target);
        }
        // If the target goes reverse, return None
        if target.start > target.end {
            return None;
        }

        // If we target 0..0 on an empty sequence, we want to expand to the entire equivalent
        if original && sequence.original.is_empty() && target == (0..0) {
            return Some(0..len_normalized);
        }
        if !original && sequence.normalized.is_empty() && target == (0..0) {
            return Some(0..len_original);
        }

        if original {
            let (mut start, mut end) = (None, None);
            sequence
                .alignments
                .iter()
                .enumerate()
                .take_while(|(_, alignment)| target.end >= alignment.1)
                .for_each(|(i, alignment)| {
                    if start.is_none() && target.start <= alignment.0 {
                        // For now, don't update if width == 0
                        if alignment.0 != alignment.1 {
                            start = Some(i);
                        }
                    }
                    if target.end >= alignment.1 {
                        end = Some(i + 1);
                    }
                });

            match (start, end) {
                // Targetting inexistant beginning
                (Some(s), None) => Some(s..s),
                // Targetting inexistant end
                (None, Some(e)) => Some(e..e),
                // Found the range
                (Some(s), Some(e)) => Some(s..e),
                _ => None,
            }
        } else {
            sequence
                .alignments
                .get(target)
                .map(Offsets::expand_alignments)
                .flatten()
        }
    }
}

/// The behavior for the delimiter of a split pattern.
///
/// # Examples
/// When splitting `the-final--countdown` on `'-'`:
/// - Remove: `[ "the", "final", "countdown" ]`.
/// - Isolate: `[ "the", "-", "final", "-", "-", "countdown" ]`.
pub enum SplitDelimiter {
    Remove,
    Isolate,
}

/// A normalized sequence.
///
/// Keeps both the original and the normalized sequence, alignment information between both and
/// provides an interface to retrieve ranges of each sequence, using offsets from any of them.
#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
pub struct NormalizedString {
    /// The original sequence.
    pub original: String,
    /// The normalized sequence.
    pub normalized: String,
    /// Byte mappings from the normalized to the original sequence.
    pub alignments: Vec<Offsets>,
    /// The offset of this sequence in a super-sequence.
    pub offset: usize,
}

impl<S> From<S> for NormalizedString
where
    S: AsRef<str>,
{
    fn from(sequence: S) -> Self {
        let sequence = sequence.as_ref();
        let alignments = sequence
            .char_indices()
            .flat_map(|(idx, chr)| {
                let len = chr.len_utf8();
                std::iter::repeat(Offsets(idx, idx + len)).take(len)
            })
            .collect::<Vec<_>>();
        Self {
            original: sequence.to_string(),
            normalized: sequence.to_string(),
            alignments,
            offset: 0,
        }
    }
}

impl NormalizedString {
    /// Byte-slices into the original sequence.
    fn slice_original(&self, range: Range<impl RangeBounds<usize>>) -> Option<&str> {
        match range {
            Range::Original(_) => self.original.get(range.into_range(self.original.len())),
            Range::Normalized(_) => self.original.get(range.convert(self)?),
        }
    }

    /// Byte-slices into the normalized sequence.
    fn slice_normalized(&self, range: Range<impl RangeBounds<usize>>) -> Option<&str> {
        match range {
            Range::Original(_) => self.normalized.get(range.convert(self)?),
            Range::Normalized(_) => self.normalized.get(range.into_range(self.normalized.len())),
        }
    }

    /// Char-slices into the normalized sequence.
    pub fn slice_char(&self, range: Range<impl RangeBounds<usize>>) -> Option<NormalizedString> {
        let (range, original_range, normalized_range) = match range {
            Range::Original(_) => {
                let range = range.into_range(self.original.len());
                let range = if self.original.is_char_boundary(range.start)
                    && self.original.is_char_boundary(range.end)
                {
                    Range::Original(range)
                } else {
                    return None;
                };
                let original_range = range.inner().clone();
                let normalized_range = range.clone().convert(self)?;
                (range, original_range, normalized_range)
            }
            Range::Normalized(_) => {
                let range = range.into_range(self.normalized.len());
                let range = if self.normalized.is_char_boundary(range.start)
                    && self.normalized.is_char_boundary(range.end)
                {
                    Range::Normalized(range)
                } else {
                    return None;
                };
                let original_range = range.clone().convert(self)?;
                let normalized_range = range.inner().clone();
                (range, original_range, normalized_range)
            }
        };

        let original = self
            .slice_original(range.clone())
            .unwrap_or_default()
            .into();
        let normalized = self.slice_normalized(range).unwrap_or_default().into();
        let alignments = self
            .alignments
            .get(normalized_range)?
            .iter()
            .map(|Offsets(start, end)| {
                Offsets(*start - original_range.start, *end - original_range.start)
            })
            .collect();
        let offset = self.offset + original_range.start;

        Some(Self {
            original,
            normalized,
            alignments,
            offset,
        })
    }

    /// Transforms the normalized sequence within the range and updates the alignments.
    ///
    /// Expects an iterator yielding each character of the new normalized sequence with a change of:
    /// - `1` if this is a new char.
    /// - `-N` if the character is right before N removed characters.
    /// - `0` if the character is replacing the existing one.
    ///
    /// The offset represents the number of potentially removed characters at the beginning of this
    /// normalized sequence.
    fn transform_within(
        mut self,
        range: Range<impl RangeBounds<usize>>,
        iter: impl IntoIterator<Item = (char, isize)>,
        offset: usize,
    ) -> Self {
        let n_range = match range {
            Range::Normalized(_) => range.into_range(self.normalized.len()),
            Range::Original(_) => match range.convert(&self) {
                Some(range) => range,
                None => return self,
            },
        };

        // Retrieve the original characters that are being replaced. This let us
        // compute the change in byte sizes along the way.
        let mut replaced_normalized = self.normalized[n_range.clone()]
            .chars()
            .collect::<Vec<_>>()
            .into_iter();
        let initial_removed: usize = (&mut replaced_normalized)
            .take(offset)
            .map(|c| c.len_utf8())
            .sum();

        let mut offset = (initial_removed + n_range.start) as isize;
        let mut alignments = Vec::with_capacity(n_range.len());
        let normalized = iter
            .into_iter()
            .map(|(c, changes)| {
                let idx = offset as usize;
                let align = if changes.is_positive() {
                    if idx < 1 {
                        Offsets(0, 0)
                    } else {
                        // This is a newly inserted character, so it shares the same alignment
                        // than the previous one
                        self.alignments[idx - 1]
                    }
                } else {
                    self.alignments[idx]
                };

                // If we are replacing a character, find it and compute the change in size
                let replaced_char = if !changes.is_positive() {
                    replaced_normalized.next()
                } else {
                    None
                };
                let replaced_char_size = replaced_char.map_or(0, |c| c.len_utf8());

                // If we are removing some characters, find them too
                let total_bytes_to_remove = if changes.is_negative() {
                    (&mut replaced_normalized)
                        .take(-changes as usize)
                        .map(|c| c.len_utf8())
                        .sum()
                } else {
                    0
                };

                // Keep track of the changes for next offsets
                offset += replaced_char_size as isize;
                offset += total_bytes_to_remove as isize;

                alignments.extend((0..c.len_utf8()).map(|_| align));

                // Then we keep only the char for string reconstruction
                c
            })
            .collect::<String>();

        self.alignments.splice(n_range.clone(), alignments);
        unsafe {
            self.normalized
                .as_mut_vec()
                .splice(n_range, normalized.bytes());
        }

        self
    }

    /// Transforms the normalized sequence and updates the alignments.
    ///
    /// Expects an iterator yielding each character of the new normalized sequence with a change of:
    /// - `1` if this is a new char.
    /// - `-N` if the character is right before N removed characters.
    /// - `0` if the character is replacing the existing one.
    ///
    /// The offset represents the number of potentially removed characters at the beginning of this
    /// normalized sequence.
    pub fn transform(self, iter: impl IntoIterator<Item = (char, isize)>, offset: usize) -> Self {
        self.transform_within(Range::Original(..), iter, offset)
    }

    /// Normalizes to NFD.
    pub fn nfd(self) -> Self {
        let normalized = self.normalized.clone();
        self.transform(normalized.nfd(), 0)
    }

    /// Filters characters.
    pub fn filter(self, keep: impl Fn(char) -> bool) -> Self {
        let mut removed: isize = 0;
        let mut removed_start: usize = 0;

        let mut transforms = Vec::with_capacity(self.normalized.len());
        let mut last_c = None;
        for c in self.normalized.chars() {
            if keep(c) {
                match last_c {
                    Some(lc) => {
                        transforms.push((lc, -removed));
                    }
                    None => {
                        removed_start = removed as usize;
                    }
                }
                last_c = Some(c);
                removed = 0;
            } else {
                removed += 1;
            }
        }
        if let Some(lc) = last_c {
            transforms.push((lc, -removed));
        }
        self.transform(transforms, removed_start)
    }

    /// Maps characters.
    pub fn map(self, f: impl Fn(char) -> char) -> Self {
        let transformations = self
            .normalized
            .chars()
            .map(|c| (f(c), 0))
            .collect::<Vec<_>>();
        self.transform(transformations, 0)
    }

    /// Calls the function for each character.
    pub fn for_each_char(&self, f: impl FnMut(char)) {
        self.normalized.chars().for_each(f);
    }

    /// Lowercases.
    pub fn lowercase(self) -> Self {
        let mut new_chars: Vec<(char, isize)> = vec![];
        self.for_each_char(|c| {
            c.to_lowercase().enumerate().for_each(|(index, c)| {
                new_chars.push((c, if index > 0 { 1 } else { 0 }));
            })
        });
        self.transform(new_chars, 0)
    }

    /// Splits wrt the pattern and handles the delimiter.
    pub fn split(
        &self,
        pattern: impl Pattern,
        behavior: SplitDelimiter,
    ) -> Result<Vec<NormalizedString>, Error> {
        let matches = pattern.find_matches(&self.normalized)?;

        // Process the matches according to the selected behavior: Vec<(Offsets, should_remove)>
        let splits = match behavior {
            SplitDelimiter::Remove => matches,
            SplitDelimiter::Isolate => matches
                .into_iter()
                .map(|(offsets, _)| (offsets, false))
                .collect(),
        };
        // Then we split according to the computed splits
        let splits = splits
            .into_iter()
            .filter_map(|(offsets, remove)| {
                if !remove {
                    Some(
                        self.slice_char(Range::Normalized(offsets.0..offsets.1))
                            .expect("NormalizedString bad split"),
                    )
                } else {
                    None
                }
            })
            .collect();

        Ok(splits)
    }
}

#[cfg(test)]
mod tests {
    use unicode_categories::UnicodeCategories;

    use super::*;

    impl NormalizedString {
        /// Recalculate original alignments
        fn alignments_original(&self) -> Vec<Offsets> {
            // Start, end are in alignments
            // offset, length are in alignments_original
            let mut alignments_original = Vec::with_capacity(self.original.len());

            // Eventual gap before first group
            let start = self.alignments[0].0;
            if start != 0 {
                alignments_original.extend(vec![Offsets(0, 0); start]);
            }

            let mut last = &self.alignments[0];
            let mut offset = 0;
            let mut length = 0;
            for alignment in &self.alignments {
                if last == alignment {
                    // This is the same group
                    length += 1;
                } else {
                    // This is a new group
                    let start = alignment.0;
                    if start < last.1 {
                        panic!("We can't have overlapping ranges.");
                    }

                    // Add the old group
                    alignments_original
                        .extend(vec![Offsets(offset, offset + length); last.1 - last.0]);
                    offset += length;
                    length = 1;

                    // Eventual gap between the 2 groups
                    alignments_original.extend(vec![Offsets(offset, offset); start - last.1]);
                }

                last = alignment;
            }
            // Add the last group
            alignments_original.extend(vec![Offsets(offset, offset + length); last.1 - last.0]);

            // Add eventual last gap
            offset += length;
            alignments_original.extend(vec![
                Offsets(offset, offset);
                self.original.len() - alignments_original.len()
            ]);

            // assert_eq!(alignments_original.len(), self.original.len());
            alignments_original
        }
    }

    #[test]
    fn test_nfd_adds_new_chars() {
        let normalized = NormalizedString::from("élégant").nfd();
        assert_eq!(
            normalized.alignments,
            &[
                Offsets(0, 2),
                Offsets(0, 2),
                Offsets(0, 2),
                Offsets(2, 3),
                Offsets(3, 5),
                Offsets(3, 5),
                Offsets(3, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 8),
                Offsets(8, 9),
            ],
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 3),
                Offsets(0, 3),
                Offsets(3, 4),
                Offsets(4, 7),
                Offsets(4, 7),
                Offsets(7, 8),
                Offsets(8, 9),
                Offsets(9, 10),
                Offsets(10, 11),
            ],
        );
    }

    #[test]
    fn test_remove_chars_added_by_nfd() {
        let normalized = NormalizedString::from("élégant")
            .nfd()
            .filter(|c| !c.is_mark_nonspacing());
        assert_eq!(normalized.normalized, "elegant");
        assert_eq!(
            normalized.alignments,
            &[
                Offsets(0, 2),
                Offsets(2, 3),
                Offsets(3, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 8),
                Offsets(8, 9),
            ],
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 1),
                Offsets(0, 1),
                Offsets(1, 2),
                Offsets(2, 3),
                Offsets(2, 3),
                Offsets(3, 4),
                Offsets(4, 5),
                Offsets(5, 6),
                Offsets(6, 7),
            ],
        );
    }

    #[test]
    fn test_remove_chars() {
        let normalized = NormalizedString::from("élégant").filter(|c| c != 'n');
        assert_eq!(normalized.normalized, "élégat");
        assert_eq!(
            normalized.alignments,
            &[
                Offsets(0, 2),
                Offsets(0, 2),
                Offsets(2, 3),
                Offsets(3, 5),
                Offsets(3, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                // Skipped range
                Offsets(8, 9),
            ],
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 2),
                Offsets(0, 2),
                Offsets(2, 3),
                Offsets(3, 5),
                Offsets(3, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 7), // Eaten n
                Offsets(7, 8),
            ],
        );
    }

    #[test]
    fn test_mixed_addition_and_removal() {
        let normalized = NormalizedString::from("élégant")
            .nfd()
            .filter(|c| !c.is_mark_nonspacing() && c != 'n');
        assert_eq!(normalized.normalized, "elegat");
        assert_eq!(
            normalized.alignments,
            &[
                Offsets(0, 2),
                Offsets(2, 3),
                Offsets(3, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(8, 9),
            ],
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 1),
                Offsets(0, 1),
                Offsets(1, 2),
                Offsets(2, 3),
                Offsets(2, 3),
                Offsets(3, 4), // g
                Offsets(4, 5), // a
                Offsets(5, 5), // Eaten n
                Offsets(5, 6),
            ],
        );
    }

    #[test]
    fn test_range_conversion() {
        let normalized = NormalizedString::from("    __Hello__   ")
            .filter(|c| !c.is_whitespace())
            .lowercase();
        let range = Range::Original(6..11).convert(&normalized).unwrap();
        assert_eq!(range, 2..7);
        assert_eq!(
            normalized.slice_normalized(Range::Normalized(range.clone())),
            Some("hello"),
        );
        assert_eq!(
            normalized.slice_original(Range::Normalized(range)),
            Some("Hello"),
        );
        assert_eq!(
            normalized.slice_normalized(Range::Original(6..11)),
            Some("hello")
        );
        assert_eq!(
            normalized.slice_original(Range::Original(6..11)),
            Some("Hello"),
        );

        // Make sure we get None only in specific cases
        assert_eq!(Range::Original(0..0).convert(&normalized), Some(0..0));
        assert_eq!(Range::Original(3..3).convert(&normalized), Some(3..3));
        assert_eq!(Range::Original(15..).convert(&normalized), Some(9..9));
        assert_eq!(Range::Original(16..).convert(&normalized), Some(16..16));
        assert_eq!(Range::Original(17..).convert(&normalized), None);
        assert_eq!(Range::Normalized(0..0).convert(&normalized), Some(0..0));
        assert_eq!(Range::Normalized(3..3).convert(&normalized), Some(3..3));
        assert_eq!(Range::Normalized(9..).convert(&normalized), Some(9..9));
        assert_eq!(Range::Normalized(10..).convert(&normalized), None);
    }

    #[test]
    fn test_original_range() {
        let normalized = NormalizedString::from("Hello_______ World!")
            .filter(|c| c != '_')
            .lowercase();
        assert_eq!(
            normalized
                .slice_normalized(Range::Normalized(6..11))
                .unwrap(),
            "world",
        );
        assert_eq!(
            normalized.slice_original(Range::Normalized(6..11)).unwrap(),
            "World",
        );
        let original_range =
            Range::Original(Range::Normalized(6..11).convert(&normalized).unwrap());
        assert_eq!(
            normalized.slice_normalized(original_range.clone()).unwrap(),
            "world",
        );
        assert_eq!(
            normalized.slice_original(original_range.clone()).unwrap(),
            "World",
        );
        assert_eq!(original_range.into_range(normalized.original.len()), 13..18,);
    }

    #[test]
    fn test_added_around_edges() {
        let normalized = NormalizedString::from("Hello").transform(
            vec![
                (' ', 1),
                ('H', 0),
                ('e', 0),
                ('l', 0),
                ('l', 0),
                ('o', 0),
                (' ', 1),
            ]
            .into_iter(),
            0,
        );

        assert_eq!(normalized.normalized, " Hello ");
        assert_eq!(
            normalized.slice_original(Range::Normalized(1..normalized.normalized.len() - 1)),
            Some("Hello"),
        );
    }

    #[test]
    fn test_added_characters_alignment() {
        let sequence = "野口 No";
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
        assert_eq!(
            normalized,
            NormalizedString {
                original: "野口 No".into(),
                normalized: " 野  口  No".into(),
                alignments: vec![
                    Offsets(0, 3),
                    Offsets(0, 3),
                    Offsets(0, 3),
                    Offsets(0, 3),
                    Offsets(0, 3),
                    Offsets(3, 6),
                    Offsets(3, 6),
                    Offsets(3, 6),
                    Offsets(3, 6),
                    Offsets(3, 6),
                    Offsets(6, 7),
                    Offsets(7, 8),
                    Offsets(8, 9),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 5),
                Offsets(0, 5),
                Offsets(0, 5),
                Offsets(5, 10),
                Offsets(5, 10),
                Offsets(5, 10),
                Offsets(10, 11),
                Offsets(11, 12),
                Offsets(12, 13),
            ],
        );
    }

    #[test]
    fn test_remove_at_beginning() {
        let normalized = NormalizedString::from("     Hello").filter(|c| !c.is_whitespace());
        assert_eq!(
            normalized.slice_original(Range::Normalized(1.."Hello".len())),
            Some("ello"),
        );
        assert_eq!(
            normalized.slice_original(Range::Normalized(0..normalized.normalized.len())),
            Some("Hello"),
        );
    }

    #[test]
    fn test_remove_at_end() {
        let normalized = NormalizedString::from("Hello    ").filter(|c| !c.is_whitespace());
        assert_eq!(
            normalized.slice_original(Range::Normalized(0..4)),
            Some("Hell"),
        );
        assert_eq!(
            normalized.slice_original(Range::Normalized(0..normalized.normalized.len())),
            Some("Hello"),
        );
    }

    #[test]
    fn test_removed_around_both_edges() {
        let normalized = NormalizedString::from("  Hello  ").filter(|c| !c.is_whitespace());
        assert_eq!(normalized.normalized, "Hello");
        assert_eq!(
            normalized.slice_original(Range::Normalized(0.."Hello".len())),
            Some("Hello"),
        );
        assert_eq!(
            normalized.slice_original(Range::Normalized(1.."Hell".len())),
            Some("ell"),
        );
    }

    #[test]
    fn test_slice() {
        let normalized = NormalizedString::from("Good Morning");
        let slice = normalized.slice_char(Range::Original(..)).unwrap();
        assert_eq!(slice.slice_original(Range::Normalized(0..4)), Some("Good"),);
        let slice = normalized.slice_char(Range::Normalized(..)).unwrap();
        assert_eq!(slice.slice_original(Range::Normalized(0..4)), Some("Good"),);
        let slice = normalized.slice_char(Range::Original(1..)).unwrap();
        assert_eq!(slice.slice_original(Range::Normalized(0..3)), Some("ood"),);

        let normalized = NormalizedString::from("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘");
        let slice = normalized.slice_char(Range::Original(0..4)).unwrap();
        assert_eq!(slice.normalized, "𝔾");
        assert_eq!(slice.original, "𝔾");
        let slice = normalized.slice_char(Range::Normalized(0..16)).unwrap();
        assert_eq!(slice.normalized, "𝔾𝕠𝕠𝕕");
        assert_eq!(slice.original, "𝔾𝕠𝕠𝕕");
    }

    #[test]
    fn test_split() {
        let normalized = NormalizedString::from("The-final--countdown")
            .split('-', SplitDelimiter::Remove)
            .unwrap();
        assert_eq!(
            normalized
                .iter()
                .map(|normalized| normalized.normalized.as_str())
                .collect::<Vec<_>>(),
            vec!["The", "final", "countdown"],
        );

        let normalized = NormalizedString::from("The-final--countdown")
            .split('-', SplitDelimiter::Isolate)
            .unwrap();
        assert_eq!(
            normalized
                .iter()
                .map(|normalized| normalized.normalized.as_str())
                .collect::<Vec<_>>(),
            vec!["The", "-", "final", "-", "-", "countdown"],
        );
    }

    #[test]
    fn test_transform_range_single_bytes() {
        let sequence = "Hello friend";

        // Removing at the beginning
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(0..4),
            vec![('Y', 0)],
            3,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Yo friend".into(),
                alignments: vec![
                    Offsets(3, 4),
                    Offsets(4, 5),
                    Offsets(5, 6),
                    Offsets(6, 7),
                    Offsets(7, 8),
                    Offsets(8, 9),
                    Offsets(9, 10),
                    Offsets(10, 11),
                    Offsets(11, 12),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 0),
                Offsets(0, 0),
                Offsets(0, 0),
                Offsets(0, 1),
                Offsets(1, 2),
                Offsets(2, 3),
                Offsets(3, 4),
                Offsets(4, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 8),
                Offsets(8, 9),
            ],
        );

        // Removing in the middle
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(3..10),
            vec![('_', 0), ('F', 0), ('R', -2)],
            2,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Hel_FRnd".into(),
                alignments: vec![
                    Offsets(0, 1),
                    Offsets(1, 2),
                    Offsets(2, 3),
                    Offsets(5, 6),
                    Offsets(6, 7),
                    Offsets(7, 8),
                    Offsets(10, 11),
                    Offsets(11, 12),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 1),
                Offsets(1, 2),
                Offsets(2, 3),
                Offsets(3, 3),
                Offsets(3, 3),
                Offsets(3, 4),
                Offsets(4, 5),
                Offsets(5, 6),
                Offsets(6, 6),
                Offsets(6, 6),
                Offsets(6, 7),
                Offsets(7, 8),
            ],
        );

        // Removing at the end
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(5..),
            vec![('_', 0), ('F', -5)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Hello_F".into(),
                alignments: vec![
                    Offsets(0, 1),
                    Offsets(1, 2),
                    Offsets(2, 3),
                    Offsets(3, 4),
                    Offsets(4, 5),
                    Offsets(5, 6),
                    Offsets(6, 7),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 1),
                Offsets(1, 2),
                Offsets(2, 3),
                Offsets(3, 4),
                Offsets(4, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 7),
                Offsets(7, 7),
                Offsets(7, 7),
                Offsets(7, 7),
                Offsets(7, 7),
            ],
        );

        // Adding at the beginning
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(0..1),
            vec![('H', 1), ('H', 0)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "HHello friend".into(),
                alignments: vec![
                    Offsets(0, 0),
                    Offsets(0, 1),
                    Offsets(1, 2),
                    Offsets(2, 3),
                    Offsets(3, 4),
                    Offsets(4, 5),
                    Offsets(5, 6),
                    Offsets(6, 7),
                    Offsets(7, 8),
                    Offsets(8, 9),
                    Offsets(9, 10),
                    Offsets(10, 11),
                    Offsets(11, 12),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(1, 2),
                Offsets(2, 3),
                Offsets(3, 4),
                Offsets(4, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 8),
                Offsets(8, 9),
                Offsets(9, 10),
                Offsets(10, 11),
                Offsets(11, 12),
                Offsets(12, 13),
            ],
        );

        // Equivalent to the previous one
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(0..0),
            vec![('H', 1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "HHello friend".into(),
                alignments: vec![
                    Offsets(0, 0),
                    Offsets(0, 1),
                    Offsets(1, 2),
                    Offsets(2, 3),
                    Offsets(3, 4),
                    Offsets(4, 5),
                    Offsets(5, 6),
                    Offsets(6, 7),
                    Offsets(7, 8),
                    Offsets(8, 9),
                    Offsets(9, 10),
                    Offsets(10, 11),
                    Offsets(11, 12),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(1, 2),
                Offsets(2, 3),
                Offsets(3, 4),
                Offsets(4, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 8),
                Offsets(8, 9),
                Offsets(9, 10),
                Offsets(10, 11),
                Offsets(11, 12),
                Offsets(12, 13),
            ],
        );

        // Adding as part of the first character
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(0..1),
            vec![('H', 0), ('H', 1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "HHello friend".into(),
                alignments: vec![
                    Offsets(0, 1),
                    Offsets(0, 1),
                    Offsets(1, 2),
                    Offsets(2, 3),
                    Offsets(3, 4),
                    Offsets(4, 5),
                    Offsets(5, 6),
                    Offsets(6, 7),
                    Offsets(7, 8),
                    Offsets(8, 9),
                    Offsets(9, 10),
                    Offsets(10, 11),
                    Offsets(11, 12),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 2),
                Offsets(2, 3),
                Offsets(3, 4),
                Offsets(4, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 8),
                Offsets(8, 9),
                Offsets(9, 10),
                Offsets(10, 11),
                Offsets(11, 12),
                Offsets(12, 13),
            ],
        );

        // Adding in the middle
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(5..6),
            vec![('_', 0), ('m', 1), ('y', 1), ('_', 1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Hello_my_friend".into(),
                alignments: vec![
                    Offsets(0, 1),
                    Offsets(1, 2),
                    Offsets(2, 3),
                    Offsets(3, 4),
                    Offsets(4, 5),
                    Offsets(5, 6),
                    Offsets(5, 6),
                    Offsets(5, 6),
                    Offsets(5, 6),
                    Offsets(6, 7),
                    Offsets(7, 8),
                    Offsets(8, 9),
                    Offsets(9, 10),
                    Offsets(10, 11),
                    Offsets(11, 12),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 1),
                Offsets(1, 2),
                Offsets(2, 3),
                Offsets(3, 4),
                Offsets(4, 5),
                Offsets(5, 9),
                Offsets(9, 10),
                Offsets(10, 11),
                Offsets(11, 12),
                Offsets(12, 13),
                Offsets(13, 14),
                Offsets(14, 15),
            ],
        );

        // Adding at the end
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(11..),
            vec![('d', 0), ('_', 1), ('!', 1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Hello friend_!".into(),
                alignments: vec![
                    Offsets(0, 1),
                    Offsets(1, 2),
                    Offsets(2, 3),
                    Offsets(3, 4),
                    Offsets(4, 5),
                    Offsets(5, 6),
                    Offsets(6, 7),
                    Offsets(7, 8),
                    Offsets(8, 9),
                    Offsets(9, 10),
                    Offsets(10, 11),
                    Offsets(11, 12),
                    Offsets(11, 12),
                    Offsets(11, 12),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 1),
                Offsets(1, 2),
                Offsets(2, 3),
                Offsets(3, 4),
                Offsets(4, 5),
                Offsets(5, 6),
                Offsets(6, 7),
                Offsets(7, 8),
                Offsets(8, 9),
                Offsets(9, 10),
                Offsets(10, 11),
                Offsets(11, 14),
            ],
        );
    }

    #[test]
    fn test_transform_range_multiple_bytes() {
        let sequence = "𝔾𝕠𝕠𝕕";

        // Removing at the beginning
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(0..8),
            vec![('G', -1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "G𝕠𝕕".into(),
                alignments: vec![
                    Offsets(0, 4),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 1),
                Offsets(0, 1),
                Offsets(0, 1),
                Offsets(0, 1),
                Offsets(1, 1),
                Offsets(1, 1),
                Offsets(1, 1),
                Offsets(1, 1),
                Offsets(1, 5),
                Offsets(1, 5),
                Offsets(1, 5),
                Offsets(1, 5),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
            ],
        );
        assert_eq!(
            normalized.slice_normalized(Range::Original(0..8)).unwrap(),
            "G"
        );
        assert_eq!(
            normalized.slice_normalized(Range::Original(0..4)).unwrap(),
            "G"
        );
        assert_eq!(
            normalized.slice_original(Range::Original(0..4)).unwrap(),
            "𝔾",
        );
        assert_eq!(
            normalized.slice_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠",
        );

        // Removing in the middle
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(4..12),
            vec![('o', -1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾o𝕕".into(),
                alignments: vec![
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(4, 8),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 4),
                Offsets(0, 4),
                Offsets(0, 4),
                Offsets(0, 4),
                Offsets(4, 5),
                Offsets(4, 5),
                Offsets(4, 5),
                Offsets(4, 5),
                Offsets(5, 5),
                Offsets(5, 5),
                Offsets(5, 5),
                Offsets(5, 5),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
            ],
        );

        // Removing at the end
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(12..),
            vec![('d', 0), ('!', 1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾𝕠𝕠d!".into(),
                alignments: vec![
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(12, 16),
                    Offsets(12, 16),
                ],
                offset: 0,
            },
        );

        // Adding at the beginning
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(0..4),
            vec![('_', 1), ('𝔾', 0)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "_𝔾𝕠𝕠𝕕".into(),
                alignments: vec![
                    Offsets(0, 0),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(1, 5),
                Offsets(1, 5),
                Offsets(1, 5),
                Offsets(1, 5),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(9, 13),
                Offsets(9, 13),
                Offsets(9, 13),
                Offsets(9, 13),
                Offsets(13, 17),
                Offsets(13, 17),
                Offsets(13, 17),
                Offsets(13, 17),
            ],
        );
        assert_eq!(
            normalized.slice_normalized(Range::Original(0..8)).unwrap(),
            "𝔾𝕠"
        );
        assert_eq!(
            normalized.slice_normalized(Range::Original(0..4)).unwrap(),
            "𝔾"
        );
        assert_eq!(
            normalized.slice_original(Range::Original(0..4)).unwrap(),
            "𝔾",
        );
        assert_eq!(
            normalized.slice_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠",
        );

        // Equivalent to the previous one
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(0..0),
            vec![('_', 1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "_𝔾𝕠𝕠𝕕".into(),
                alignments: vec![
                    Offsets(0, 0),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(1, 5),
                Offsets(1, 5),
                Offsets(1, 5),
                Offsets(1, 5),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(9, 13),
                Offsets(9, 13),
                Offsets(9, 13),
                Offsets(9, 13),
                Offsets(13, 17),
                Offsets(13, 17),
                Offsets(13, 17),
                Offsets(13, 17),
            ],
        );
        assert_eq!(
            normalized.slice_normalized(Range::Original(0..8)).unwrap(),
            "𝔾𝕠"
        );
        assert_eq!(
            normalized.slice_normalized(Range::Original(0..4)).unwrap(),
            "𝔾"
        );
        assert_eq!(
            normalized.slice_original(Range::Original(0..4)).unwrap(),
            "𝔾",
        );
        assert_eq!(
            normalized.slice_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠",
        );

        // Adding as part of the first character
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(0..4),
            vec![('𝔾', 0), ('o', 1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾o𝕠𝕠𝕕".into(),
                alignments: vec![
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 5),
                Offsets(0, 5),
                Offsets(0, 5),
                Offsets(0, 5),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(5, 9),
                Offsets(9, 13),
                Offsets(9, 13),
                Offsets(9, 13),
                Offsets(9, 13),
                Offsets(13, 17),
                Offsets(13, 17),
                Offsets(13, 17),
                Offsets(13, 17),
            ],
        );
        assert_eq!(
            normalized.slice_normalized(Range::Original(0..8)).unwrap(),
            "𝔾o𝕠"
        );
        assert_eq!(
            normalized.slice_normalized(Range::Original(0..4)).unwrap(),
            "𝔾o"
        );
        assert_eq!(
            normalized.slice_original(Range::Original(0..4)).unwrap(),
            "𝔾",
        );
        assert_eq!(
            normalized.slice_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠",
        );

        // Adding in the middle
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(4..8),
            vec![('𝕠', 0), ('o', 1), ('o', 1), ('o', 1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾𝕠ooo𝕠𝕕".into(),
                alignments: vec![
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 4),
                Offsets(0, 4),
                Offsets(0, 4),
                Offsets(0, 4),
                Offsets(4, 11),
                Offsets(4, 11),
                Offsets(4, 11),
                Offsets(4, 11),
                Offsets(11, 15),
                Offsets(11, 15),
                Offsets(11, 15),
                Offsets(11, 15),
                Offsets(15, 19),
                Offsets(15, 19),
                Offsets(15, 19),
                Offsets(15, 19),
            ],
        );

        // Adding at the end
        let normalized = NormalizedString::from(sequence).transform_within(
            Range::Original(16..),
            vec![('!', 1)],
            0,
        );
        assert_eq!(
            normalized,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾𝕠𝕠𝕕!".into(),
                alignments: vec![
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(0, 4),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(4, 8),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(8, 12),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                    Offsets(12, 16),
                ],
                offset: 0,
            },
        );
        assert_eq!(
            normalized.alignments_original(),
            vec![
                Offsets(0, 4),
                Offsets(0, 4),
                Offsets(0, 4),
                Offsets(0, 4),
                Offsets(4, 8),
                Offsets(4, 8),
                Offsets(4, 8),
                Offsets(4, 8),
                Offsets(8, 12),
                Offsets(8, 12),
                Offsets(8, 12),
                Offsets(8, 12),
                Offsets(12, 17),
                Offsets(12, 17),
                Offsets(12, 17),
                Offsets(12, 17),
            ],
        );
    }

    #[test]
    fn test_transform_check() {
        let normalized = NormalizedString::from("abc…")
            .nfd()
            .transform(vec![('a', -2), ('…', 0)], 0)
            .lowercase();
        assert_eq!(normalized.normalized, "a…");
    }
}
