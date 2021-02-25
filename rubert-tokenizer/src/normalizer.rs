use std::{
    cmp::Ordering,
    iter,
    ops::{Bound, RangeBounds},
};

use regex::Regex;
use unicode_categories::UnicodeCategories;
use unicode_normalization_alignments::UnicodeNormalization;

use crate::Error;

/// A normalizer.
///
/// Defaults to the [`none()`] normalizer.
pub struct Normalizer(Normalizers);

/// The normalizers.
enum Normalizers {
    /// No normalization.
    None,
    /// Bert normalization.
    Bert {
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: bool,
        lowercase: bool,
    },
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::none()
    }
}

impl Normalizer {
    /// Creates an inert normalizer.
    pub fn none() -> Self {
        Self(Normalizers::None)
    }

    /// Creates a Bert normalizer.
    ///
    /// Configurable by:
    /// - `clean_text`: Removes any control characters and replaces all sorts of whitespace by ` `.
    /// - `handle_chinese_chars`: Puts spaces around chinese characters so they get split.
    /// - `strip_accents`: Removes accents from characters.
    /// - `lowercase`: Lowercases characters.
    pub fn bert(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: bool,
        lowercase: bool,
    ) -> Self {
        Self(Normalizers::Bert {
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
        })
    }

    fn clean_text(&self, normalized: NormalizedString) -> NormalizedString {
        match self.0 {
            Normalizers::None
            | Normalizers::Bert {
                clean_text: false, ..
            } => normalized,
            Normalizers::Bert {
                clean_text: true, ..
            } => normalized
                .filter(|c| {
                    c != '\0'
                        && c != '\u{fffd}'
                        && (c == '\t' || c == '\n' || c == '\r' || !c.is_other())
                })
                .map(|c| {
                    // These are technically control characters but we count them as whitespace
                    // The definition of `is_control` here is quite large and contains also
                    // Cc, Cf, Cn or Co; cf. https://unicode.org/reports/tr44/ (Table 12)
                    if c == '\t' || c == '\n' || c == '\r' || c.is_whitespace() {
                        ' '
                    } else {
                        c
                    }
                }),
        }
    }

    fn handle_chinese_chars(&self, normalized: NormalizedString) -> NormalizedString {
        match self.0 {
            Normalizers::None
            | Normalizers::Bert {
                handle_chinese_chars: false,
                ..
            } => normalized,
            Normalizers::Bert {
                handle_chinese_chars: true,
                ..
            } => {
                let mut new_chars: Vec<(char, isize)> = vec![];
                normalized.for_each(|c| {
                    // Checks whether a character is chinese
                    // This defines a "chinese character" as anything in the CJK Unicode block:
                    //   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
                    //
                    // Note that the CJK Unicode block is NOT all Japanese and Korean characters,
                    // despite its name. The modern Korean Hangul alphabet is a different block,
                    // as is Japanese Hiragana and Katakana. Those alphabets are used to write
                    // space-separated words, so they are not treated specially and handled
                    // like for all of the other languages.
                    if let '\u{4E00}'..='\u{9FFF}'
                    | '\u{3400}'..='\u{4DBF}'
                    | '\u{20000}'..='\u{2A6DF}'
                    | '\u{2A700}'..='\u{2B73F}'
                    | '\u{2B740}'..='\u{2B81F}'
                    | '\u{2B920}'..='\u{2CEAF}'
                    | '\u{F900}'..='\u{FAFF}'
                    | '\u{2F800}'..='\u{2FA1F}' = c
                    {
                        new_chars.extend(&[(' ', 0), (c, 1), (' ', 1)]);
                    } else {
                        new_chars.push((c, 0));
                    }
                });
                normalized.transform(new_chars, 0)
            }
        }
    }

    fn strip_accents(&self, normalized: NormalizedString) -> NormalizedString {
        match self.0 {
            Normalizers::None
            | Normalizers::Bert {
                strip_accents: false,
                ..
            } => normalized,
            Normalizers::Bert {
                strip_accents: true,
                ..
            } => normalized.nfd().filter(|c| !c.is_mark_nonspacing()),
        }
    }

    fn lowercase(&self, normalized: NormalizedString) -> NormalizedString {
        match self.0 {
            Normalizers::None
            | Normalizers::Bert {
                lowercase: false, ..
            } => normalized,
            Normalizers::Bert {
                lowercase: true, ..
            } => normalized.lowercase(),
        }
    }

    pub(crate) fn normalize(&self, sequence: impl Into<NormalizedString>) -> NormalizedString {
        let normalized = self.clean_text(sequence.into());
        let normalized = self.handle_chinese_chars(normalized);
        let normalized = self.strip_accents(normalized);
        self.lowercase(normalized)
    }
}

/// The possible offsets referential.
pub enum OffsetReferential {
    Original,
    Normalized,
}

/// Represents a Range usable by the NormalizedString to index its content.
/// A Range can use indices relative to either the `Original` or the `Normalized` string
#[derive(Clone)]
pub enum Range<T> {
    Original(T),
    Normalized(T),
}

impl<T> Range<T>
where
    T: RangeBounds<usize> + Clone,
{
    fn inner(&self) -> &T {
        match self {
            Range::Original(range) => range,
            Range::Normalized(range) => range,
        }
    }

    /// Return the length of the current Range if not Unbounded
    fn len(&self) -> Option<usize> {
        let range = self.inner();
        let end = match range.end_bound() {
            Bound::Included(idx) => *idx + 1,
            Bound::Excluded(idx) => *idx,
            Bound::Unbounded => return None,
        };
        match range.start_bound() {
            Bound::Included(idx) => Some(end - (*idx + 1)),
            Bound::Excluded(idx) => Some(end - *idx),
            Bound::Unbounded => Some(end),
        }
    }

    /// Converts the current Range to a `std::ops::Range<usize>`. This requires the `max_len`
    /// of the represented string (in chars, not bytes) in order to cover the case where the
    /// original provided range was unbounded
    fn into_full_range(self, max_len: usize) -> std::ops::Range<usize> {
        let range = self.inner();
        let start = match range.start_bound() {
            Bound::Included(idx) => *idx,
            Bound::Excluded(idx) => *idx + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(idx) => *idx + 1,
            Bound::Excluded(idx) => *idx,
            Bound::Unbounded => max_len,
        };

        start..end
    }
}

/// Defines the expected behavior for the delimiter of a Split Pattern.
///
/// # Examples
/// When splitting on `'-'` with input `the-final--countdown`:
/// - Removed: `[ "the", "final", "countdown" ]`
/// - Isolated: `[ "the", "-", "final", "-", "-", "countdown" ]`
/// - MergedWithPrevious: `[ "the-", "final-", "-", "countdown" ]`
/// - MergedWithNext: `[ "the", "-final", "-", "-countdown" ]`
/// - Contiguous: `[ "the", "-", "final", "--", "countdown" ]`
pub enum SplitDelimiterBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

/// A `NormalizedString` takes care of processing an "original" string to modify
/// it and obtain a "normalized" string. It keeps both version of the string,
/// alignments information between both and provides an interface to retrieve
/// ranges of each string, using offsets from any of them.
///
/// It is possible to retrieve a part of the original string, by indexing it with
/// offsets from the normalized one, and the other way around too. It is also
/// possible to convert offsets from one referential to the other one easily.
#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
pub struct NormalizedString {
    /// The original version of the string, before any modification
    pub(crate) original: String,
    /// The normalized version of the string, after all modifications
    pub(crate) normalized: String,
    /// Mapping from normalized string to original one: (start, end) for each
    /// byte of the normalized string
    pub(crate) alignments: Vec<Offsets>,
    /// If this NormalizedString is a slice of a bigger one, we keep the track
    /// of the missing part, so that we can still give offsets from this original
    /// string.
    pub(crate) original_shift: usize,
}

impl NormalizedString {
    /// Returns the original offsets.
    pub(crate) fn offsets_original(&self) -> Offsets {
        Offsets(
            self.original_shift,
            self.original_shift + self.original.len(),
        )
    }

    /// Convert the given offsets range from one referential to the other one:
    /// `Original => Normalized` or `Normalized => Original`
    ///
    /// Returns `None` when targeting something that is outside range
    pub(crate) fn convert_offsets<T>(&self, range: Range<T>) -> Option<std::ops::Range<usize>>
    where
        T: RangeBounds<usize> + Clone,
    {
        let len_original = self.original.len();
        let len_normalized = self.normalized.len();

        let (target, original) = match range {
            Range::Original(_) => (range.into_full_range(len_original), true),
            Range::Normalized(_) => (range.into_full_range(len_normalized), false),
        };

        // If we target an empty range, let's return the same
        if target.start == target.end {
            return Some(target);
        }
        // If the target goes reverse, return None
        if target.start > target.end {
            return None;
        }

        // If we target 0..0 on an empty string, we want to expand to the entire equivalent
        if original && self.original.is_empty() && target == (0..0) {
            return Some(0..len_normalized);
        }
        if !original && self.normalized.is_empty() && target == (0..0) {
            return Some(0..len_original);
        }

        if original {
            let (mut start, mut end) = (None, None);
            self.alignments
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
            self.alignments.get(target).map(expand_alignments).flatten()
        }
    }

    /// Return a range of the normalized string
    fn get_range<T>(&self, range: Range<T>) -> Option<&str>
    where
        T: RangeBounds<usize> + Clone,
    {
        match range {
            Range::Original(_) => self.normalized.get(self.convert_offsets(range)?),
            Range::Normalized(_) => self
                .normalized
                .get(range.into_full_range(self.normalized.len())),
        }
    }

    /// Return a range of the original string
    fn get_range_original<T>(&self, range: Range<T>) -> Option<&str>
    where
        T: RangeBounds<usize> + Clone,
    {
        match range {
            Range::Original(_) => self
                .original
                .get(range.into_full_range(self.original.len())),
            Range::Normalized(_) => self.original.get(self.convert_offsets(range)?),
        }
    }

    /// Validate the given range, to make sure it is on char boundaries
    fn validate_range<T: RangeBounds<usize> + Clone>(
        &self,
        range: Range<T>,
    ) -> Option<Range<std::ops::Range<usize>>> {
        match range {
            Range::Original(_) => {
                let r = range.into_full_range(self.original.len());
                if !(self.original.is_char_boundary(r.start)
                    && self.original.is_char_boundary(r.end))
                {
                    None
                } else {
                    Some(Range::Original(r))
                }
            }
            Range::Normalized(_) => {
                let r = range.into_full_range(self.normalized.len());
                if !(self.normalized.is_char_boundary(r.start)
                    && self.normalized.is_char_boundary(r.end))
                {
                    None
                } else {
                    Some(Range::Normalized(r))
                }
            }
        }
    }

    /// Return a slice of the current NormalizedString
    /// If the range is not on char boundaries, return None
    pub(crate) fn slice<T>(&self, range: Range<T>) -> Option<NormalizedString>
    where
        T: RangeBounds<usize> + Clone,
    {
        let full_range = self.validate_range(range)?;
        let (normalized_range, original_range) = match full_range {
            Range::Original(_) => (
                self.convert_offsets(full_range.clone())?,
                full_range.inner().clone(),
            ),
            Range::Normalized(_) => (
                full_range.inner().clone(),
                self.convert_offsets(full_range.clone())?,
            ),
        };

        let n_shift = original_range.start;

        Some(Self {
            original: self
                .get_range_original(full_range.clone())
                .unwrap_or_default()
                .into(),
            normalized: self.get_range(full_range).unwrap_or_default().into(),
            alignments: self
                .alignments
                .get(normalized_range)?
                .iter()
                .map(|Offsets(start, end)| Offsets(*start - n_shift, *end - n_shift))
                .collect(),
            original_shift: self.original_shift + original_range.start,
        })
    }

    /// Applies transformations to the current normalized version of the string,
    /// while updating the alignments.
    /// This method expect an Iterator yielding each char of the new normalized string
    /// with a `change` isize equals to:
    ///   - `1` if this is a new char
    ///   - `-N` if the char is right before N removed chars
    ///   - `0` if the char is replacing the existing one
    /// Since it is possible that the normalized string doesn't include some of the characters at
    /// the beginning of the original one, we need an `initial_offset` which represents the number
    /// of removed chars at the very beginning.
    fn transform_range<T, I>(mut self, range: Range<T>, dest: I, initial_offset: usize) -> Self
    where
        T: RangeBounds<usize> + Clone,
        I: IntoIterator<Item = (char, isize)>,
    {
        let n_range = match range {
            Range::Normalized(_) => range.into_full_range(self.normalized.len()),
            Range::Original(_) => match self.convert_offsets(range) {
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
            .take(initial_offset)
            .map(|c| c.len_utf8())
            .sum();

        let mut offset = (initial_removed + n_range.start) as isize;
        let mut alignments = Vec::with_capacity(n_range.len());
        let normalized = dest
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

    /// Applies transformations to the current normalized version of the string,
    /// while updating the alignments.
    /// This method expect an Iterator yielding each char of the new normalized string
    /// with a `change` isize equals to:
    ///   - `1` if this is a new char
    ///   - `-N` if the char is right before N removed chars
    ///   - `0` if the char is replacing the existing one
    /// Since it is possible that the normalized string doesn't include some of the characters at
    /// the beginning of the original one, we need an `initial_offset` which represents the number
    /// of removed chars at the very beginning.
    pub(crate) fn transform<I>(self, dest: I, initial_offset: usize) -> Self
    where
        I: IntoIterator<Item = (char, isize)>,
    {
        self.transform_range(Range::Original(..), dest, initial_offset)
    }

    /// Applies NFD normalization
    pub(crate) fn nfd(self) -> Self {
        let normalized = self.normalized.clone();
        self.transform(normalized.nfd(), 0)
    }

    /// Applies filtering over our characters
    pub(crate) fn filter<F: Fn(char) -> bool>(self, keep: F) -> Self {
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

    /// Map our characters
    pub(crate) fn map<F: Fn(char) -> char>(self, f: F) -> Self {
        let transformations = self
            .normalized
            .chars()
            .map(|c| (f(c), 0))
            .collect::<Vec<_>>();
        self.transform(transformations, 0)
    }

    /// Calls the given function for each characters
    pub(crate) fn for_each<F: FnMut(char)>(&self, foreach: F) -> &Self {
        self.normalized.chars().for_each(foreach);
        self
    }

    /// Lowercase
    pub(crate) fn lowercase(self) -> Self {
        let mut new_chars: Vec<(char, isize)> = vec![];
        self.for_each(|c| {
            c.to_lowercase().enumerate().for_each(|(index, c)| {
                new_chars.push((c, if index > 0 { 1 } else { 0 }));
            })
        });
        self.transform(new_chars, 0)
    }

    /// Replace anything that matches the pattern with the given content.
    fn replace<P: Pattern>(self, pattern: P, content: &str) -> Result<Self, Error> {
        let (this, _) = pattern
            .find_matches(self.normalized.as_str())?
            .into_iter()
            .fold(
                (self, 0_isize),
                |(this, offset), (Offsets(start, end), is_match)| {
                    if is_match {
                        let range = match offset.cmp(&0) {
                            Ordering::Less => {
                                // makes sure of avoiding any substraction overflow, flooring at 0
                                let offset = -offset as usize;
                                match (start.overflowing_sub(offset), end.overflowing_sub(offset)) {
                                    ((start @ _, false), (end @ _, false)) => start..end,
                                    ((_, true), (end @ _, false)) => 0..end,
                                    _ => 0..0,
                                }
                            }
                            Ordering::Equal => start..end,
                            Ordering::Greater => {
                                let offset = offset as usize;
                                start + offset..end + offset
                            }
                        };

                        let mut new_len = 0;
                        let removed_chars = this.normalized[range.clone()].chars().count();
                        let this = this.transform_range(
                            Range::Normalized(range),
                            content.chars().map(|c| {
                                new_len += c.len_utf8();
                                (c, 1)
                            }),
                            removed_chars,
                        );

                        let old_len = end - start;
                        let offset = offset + new_len as isize - old_len as isize;

                        (this, offset)
                    } else {
                        (this, offset)
                    }
                },
            );
        Ok(this)
    }

    /// Split the current string in many subparts. Specify what to do with the
    /// delimiter.
    ///
    /// ## Splitting Behavior for the delimiter
    ///
    /// The behavior can be one of the followings:
    /// When splitting on `'-'` for example, with input `the-final--countdown`:
    ///  - Removed => `[ "the", "", "final", "", "", "countdown" ]`
    ///  - Isolated => `[ "the", "-", "final", "-", "-", "countdown" ]`
    ///  - MergedWithPrevious => `[ "the-", "final-", "-", "countdown" ]`
    ///  - MergedWithNext => `[ "the", "-final", "-", "-countdown" ]`
    pub(crate) fn split<P: Pattern>(
        &self,
        pattern: P,
        behavior: SplitDelimiterBehavior,
    ) -> Result<Vec<NormalizedString>, Error> {
        let matches = pattern.find_matches(&self.normalized)?;

        // Process the matches according to the selected behavior: Vec<(Offsets, should_remove)>
        use SplitDelimiterBehavior::*;
        let splits = match behavior {
            Isolated => matches
                .into_iter()
                .map(|(offsets, _)| (offsets, false))
                .collect(),
            Removed => matches,
            Contiguous => {
                let mut previous_match = false;
                matches
                    .into_iter()
                    .fold(vec![], |mut acc, (offsets, is_match)| {
                        if is_match == previous_match {
                            if let Some((Offsets(_, end), _)) = acc.last_mut() {
                                *end = offsets.1;
                            } else {
                                acc.push((offsets, false));
                            }
                        } else {
                            acc.push((offsets, false));
                        }
                        previous_match = is_match;
                        acc
                    })
            }
            MergedWithPrevious => {
                let mut previous_match = false;
                matches
                    .into_iter()
                    .fold(vec![], |mut acc, (offsets, is_match)| {
                        if is_match && !previous_match {
                            if let Some((Offsets(_, end), _)) = acc.last_mut() {
                                *end = offsets.1;
                            } else {
                                acc.push((offsets, false));
                            }
                        } else {
                            acc.push((offsets, false));
                        }
                        previous_match = is_match;
                        acc
                    })
            }
            MergedWithNext => {
                let mut previous_match = false;
                let mut matches =
                    matches
                        .into_iter()
                        .rev()
                        .fold(vec![], |mut acc, (offsets, is_match)| {
                            if is_match && !previous_match {
                                if let Some((Offsets(start, _), _)) = acc.last_mut() {
                                    *start = offsets.0;
                                } else {
                                    acc.push((offsets, false));
                                }
                            } else {
                                acc.push((offsets, false));
                            }
                            previous_match = is_match;
                            acc
                        });
                matches.reverse();
                matches
            }
        };

        // Then we split according to the computed splits
        Ok(splits
            .into_iter()
            .filter_map(|(offsets, remove)| {
                if !remove {
                    Some(
                        self.slice(Range::Normalized(offsets.0..offsets.1))
                            .expect("NormalizedString bad split"),
                    )
                } else {
                    None
                }
            })
            .collect())
    }

    fn strip(self, left: bool, right: bool) -> Self {
        let leading_spaces = if left {
            self.normalized
                .chars()
                .take_while(|c| c.is_whitespace())
                .count()
        } else {
            0
        };
        let trailing_spaces = if right {
            self.normalized
                .chars()
                .rev()
                .take_while(|c| c.is_whitespace())
                .count()
        } else {
            0
        };

        if leading_spaces > 0 || trailing_spaces > 0 {
            let transformation = self
                .normalized
                .chars()
                .enumerate()
                .filter_map(|(i, c)| {
                    if i < leading_spaces || i >= self.normalized.len() - trailing_spaces {
                        None
                    } else if i == self.normalized.len() - trailing_spaces - 1 {
                        Some((c, -(trailing_spaces as isize)))
                    } else {
                        Some((c, 0))
                    }
                })
                .collect::<Vec<_>>();
            self.transform(transformation, leading_spaces)
        } else {
            self
        }
    }

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
                alignments_original.extend(vec![Offsets(offset, offset + length); last.1 - last.0]);
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

/// Returns the range covered by a slice of alignments
fn expand_alignments(alignments: &[Offsets]) -> Option<std::ops::Range<usize>> {
    if alignments.is_empty() {
        None
    } else {
        let start = alignments[0].0;
        let end = alignments[alignments.len() - 1].1;
        Some(start..end)
    }
}

/// Returns a range of the given string slice, by indexing chars instead of bytes
fn get_range_of<T: RangeBounds<usize>>(s: &str, range: T) -> Option<&str> {
    let len = s.chars().count();
    let start = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(i) => *i,
        Bound::Excluded(i) => *i + 1,
    };
    let end = match range.end_bound() {
        Bound::Unbounded => len,
        Bound::Included(i) => *i + 1,
        Bound::Excluded(i) => *i,
    };

    if start == 0 && end == 0 {
        Some(&s[0..0])
    } else if start >= len || end > len || start >= end {
        None
    } else {
        let start_b = s
            .char_indices()
            .map(|(i, _)| i)
            .nth(start as usize)
            .unwrap_or(0);
        let end_b = s
            .char_indices()
            .map(|(i, _)| i)
            .nth(end as usize)
            .unwrap_or_else(|| s.len());
        Some(&s[start_b..end_b])
    }
}

/// Convert the given range from bytes to char
fn bytes_to_char(s: &str, range: std::ops::Range<usize>) -> Option<std::ops::Range<usize>> {
    let (mut start, mut end) = if range == (0..0) {
        (Some(0), Some(0))
    } else {
        (None, None)
    };

    s.char_indices()
        .enumerate()
        .take_while(|(_, (b, _))| *b <= range.end)
        .filter(|(_, (b, _))| *b >= range.start)
        .for_each(|(i, (b, c))| {
            if b == range.start {
                start = Some(i);
            }
            if b == range.end {
                end = Some(i);
            }
            if b + c.len_utf8() == range.end {
                end = Some(i + 1);
            }
        });

    Some(start?..end?)
}

/// Convert the given range from char to bytes
fn char_to_bytes(s: &str, range: std::ops::Range<usize>) -> Option<std::ops::Range<usize>> {
    let (mut start, mut end) = if range == (0..0) {
        (Some(0), Some(0))
    } else {
        (None, None)
    };

    if range.start == range.end {
        s.char_indices()
            .skip(range.start)
            .take(1)
            .for_each(|(b, _)| {
                start = Some(b);
                end = Some(b);
            });
    } else {
        s.char_indices()
            .skip(range.start)
            .take(range.end - range.start)
            .for_each(|(b, c)| {
                if start.is_none() {
                    start = Some(b);
                }
                end = Some(b + c.len_utf8());
            });
    }

    Some(start?..end?)
}

impl From<String> for NormalizedString {
    fn from(string: String) -> Self {
        let alignments = string
            .char_indices()
            .flat_map(|(idx, chr)| {
                let len = chr.len_utf8();
                iter::repeat(Offsets(idx, idx + len)).take(len)
            })
            .collect::<Vec<_>>();
        Self {
            original: string.clone(),
            normalized: string,
            alignments,
            original_shift: 0,
        }
    }
}

impl From<&str> for NormalizedString {
    fn from(string: &str) -> Self {
        string.to_string().into()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(test, derive(Debug))]
pub struct Offsets(pub usize, pub usize);

/// Pattern used to split a NormalizedString
pub(crate) trait Pattern {
    /// Slice the given string in a list of pattern match positions, with
    /// a boolean indicating whether this is a match or not.
    ///
    /// This method *must* cover the whole string in its outputs, with
    /// contiguous ordered slices.
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>, Error>;
}

impl Pattern for char {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>, Error> {
        let is_char = |c: char| -> bool { c == *self };
        is_char.find_matches(inside)
    }
}

impl Pattern for &str {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>, Error> {
        if self.is_empty() {
            // If we try to find the matches with an empty string, just don't match anything
            return Ok(vec![(Offsets(0, inside.chars().count()), false)]);
        }

        let re = Regex::new(regex::escape(self).as_str())?;
        (&re).find_matches(inside)
    }
}

impl Pattern for String {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>, Error> {
        self.as_str().find_matches(inside)
    }
}

impl Pattern for Regex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>, Error> {
        if inside.is_empty() {
            return Ok(vec![(Offsets(0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        for m in self.find_iter(inside) {
            if prev != m.start() {
                splits.push((Offsets(prev, m.start()), false));
            }
            splits.push((Offsets(m.start(), m.end()), true));
            prev = m.end();
        }
        if prev != inside.len() {
            splits.push((Offsets(prev, inside.len()), false))
        }
        Ok(splits)
    }
}

impl Pattern for &Regex {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>, Error> {
        <Regex as Pattern>::find_matches(self, inside)
    }
}

impl<F> Pattern for F
where
    F: Fn(char) -> bool,
{
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>, Error> {
        if inside.is_empty() {
            return Ok(vec![(Offsets(0, 0), false)]);
        }

        let mut last_offset = 0;
        let mut last_seen = 0;

        let mut matches = inside
            .char_indices()
            .flat_map(|(idx, chr)| {
                last_seen = idx + chr.len_utf8();
                if self(chr) {
                    let mut events = Vec::with_capacity(2);
                    if last_offset < idx {
                        // We need to emit what was before this match
                        events.push((Offsets(last_offset, idx), false));
                    }
                    events.push((Offsets(idx, last_seen), true));
                    last_offset = last_seen;
                    events
                } else {
                    vec![]
                }
            })
            .collect::<Vec<_>>();

        // Do not forget the last potential split
        if last_seen > last_offset {
            matches.push((Offsets(last_offset, last_seen), false));
        }

        Ok(matches)
    }
}

/// Invert the `is_match` flags for the wrapped Pattern. This is usefull
/// for example when we use a regex that matches words instead of a delimiter,
/// and we want to match the delimiter.
pub struct Invert<P>(pub(crate) P);

impl<P: Pattern> Pattern for Invert<P> {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>, Error> {
        Ok(self
            .0
            .find_matches(inside)?
            .into_iter()
            .map(|(offsets, flag)| (offsets, !flag))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfd_adds_new_chars() {
        let n = NormalizedString::from("élégant").nfd();
        assert_eq!(
            &n.alignments,
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
            n.alignments_original(),
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
        let n = NormalizedString::from("élégant")
            .nfd()
            .filter(|c| !c.is_mark_nonspacing());

        assert_eq!(n.normalized, "elegant");

        assert_eq!(
            &n.alignments,
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
            n.alignments_original(),
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
        let n = NormalizedString::from("élégant").filter(|c| c != 'n');
        assert_eq!(n.normalized, "élégat");
        assert_eq!(
            &n.alignments,
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
            n.alignments_original(),
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
        let n = NormalizedString::from("élégant")
            .nfd()
            .filter(|c| !c.is_mark_nonspacing() && c != 'n');
        assert_eq!(n.normalized, "elegat");
        assert_eq!(
            &n.alignments,
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
            n.alignments_original(),
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
        let n = NormalizedString::from("    __Hello__   ")
            .filter(|c| !c.is_whitespace())
            .lowercase();
        let hello_n = n.convert_offsets(Range::Original(6..11));
        assert_eq!(hello_n, Some(2..7));
        assert_eq!(
            n.get_range(Range::Normalized(hello_n.clone().unwrap())),
            Some("hello")
        );
        assert_eq!(
            n.get_range_original(Range::Normalized(hello_n.unwrap())),
            Some("Hello")
        );
        assert_eq!(n.get_range(Range::Original(6..11)), Some("hello"));
        assert_eq!(n.get_range_original(Range::Original(6..11)), Some("Hello"));

        // Make sure we get None only in specific cases
        assert_eq!(n.convert_offsets(Range::Original(0..0)), Some(0..0));
        assert_eq!(n.convert_offsets(Range::Original(3..3)), Some(3..3));
        assert_eq!(n.convert_offsets(Range::Original(15..)), Some(9..9));
        assert_eq!(n.convert_offsets(Range::Original(16..)), Some(16..16));
        assert_eq!(n.convert_offsets(Range::Original(17..)), None);
        assert_eq!(n.convert_offsets(Range::Normalized(0..0)), Some(0..0));
        assert_eq!(n.convert_offsets(Range::Normalized(3..3)), Some(3..3));
        assert_eq!(n.convert_offsets(Range::Normalized(9..)), Some(9..9));
        assert_eq!(n.convert_offsets(Range::Normalized(10..)), None);
    }

    #[test]
    fn test_original_range() {
        let n = NormalizedString::from("Hello_______ World!")
            .filter(|c| c != '_')
            .lowercase();
        let world_n = n.get_range(Range::Normalized(6..11)).unwrap();
        let world_o = n.get_range_original(Range::Normalized(6..11)).unwrap();
        assert_eq!(world_n, "world");
        assert_eq!(world_o, "World");
        let original_range = Range::Original(n.convert_offsets(Range::Normalized(6..11)).unwrap());
        assert_eq!(n.get_range(original_range.clone()).unwrap(), "world");
        assert_eq!(
            n.get_range_original(original_range.clone()).unwrap(),
            "World"
        );
        assert_eq!(original_range.into_full_range(n.original.len()), 13..18);
    }

    #[test]
    fn test_added_around_edges() {
        let n = NormalizedString::from("Hello").transform(
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

        assert_eq!(&n.normalized, " Hello ");
        assert_eq!(
            n.get_range_original(Range::Normalized(1..n.normalized.len() - 1)),
            Some("Hello")
        );
    }

    #[test]
    fn test_added_characters_alignment() {
        let n = NormalizedString::from("野口 No");
        let normalized = n.normalized.clone();
        let n = n.transform(
            normalized.chars().flat_map(|c| {
                if (c as usize) > 0x4E00 {
                    vec![(' ', 0), (c, 1), (' ', 1)]
                } else {
                    vec![(c, 0)]
                }
            }),
            0,
        );

        assert_eq!(
            n,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            n.alignments_original(),
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
        let n = NormalizedString::from("     Hello").filter(|c| !c.is_whitespace());
        assert_eq!(
            n.get_range_original(Range::Normalized(1.."Hello".len())),
            Some("ello"),
        );
        assert_eq!(
            n.get_range_original(Range::Normalized(0..n.normalized.len())),
            Some("Hello"),
        );
    }

    #[test]
    fn test_remove_at_end() {
        let n = NormalizedString::from("Hello    ").filter(|c| !c.is_whitespace());
        assert_eq!(n.get_range_original(Range::Normalized(0..4)), Some("Hell"));
        assert_eq!(
            n.get_range_original(Range::Normalized(0..n.normalized.len())),
            Some("Hello"),
        );
    }

    #[test]
    fn test_removed_around_both_edges() {
        let n = NormalizedString::from("  Hello  ").filter(|c| !c.is_whitespace());
        assert_eq!(&n.normalized, "Hello");

        assert_eq!(
            n.get_range_original(Range::Normalized(0.."Hello".len())),
            Some("Hello"),
        );
        assert_eq!(
            n.get_range_original(Range::Normalized(1.."Hell".len())),
            Some("ell"),
        );
    }

    #[test]
    fn test_get_range() {
        let s = String::from("Hello my name is John 👋");
        assert_eq!(get_range_of(&s, ..), Some(&s[..]));
        assert_eq!(get_range_of(&s, 17..), Some("John 👋"));
    }

    #[test]
    fn test_slice() {
        let s = NormalizedString::from("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘").nfd();

        let original_slice = s.slice(Range::Original(0..4)).unwrap();
        assert_eq!(original_slice.normalized, "𝔾");
        assert_eq!(original_slice.original, "𝔾");

        let normalized_slice = s.slice(Range::Normalized(0..16)).unwrap();
        assert_eq!(normalized_slice.normalized, "𝔾𝕠𝕠𝕕");
        assert_eq!(normalized_slice.original, "𝔾𝕠𝕠𝕕");

        // Make sure the sliced NormalizedString is still aligned as expected
        let s = NormalizedString::from("   Good Morning!   ").strip(true, true);

        // If we keep the whole slice
        let slice = s.slice(Range::Original(..)).unwrap();
        assert_eq!(
            slice.get_range_original(Range::Normalized(0..4)),
            Some("Good"),
        );
        let slice = s.slice(Range::Normalized(..)).unwrap();
        assert_eq!(
            slice.get_range_original(Range::Normalized(0..4)),
            Some("Good"),
        );

        // If we keep after the modified piece
        let slice = s.slice(Range::Original(4..15)).unwrap();
        assert_eq!(
            slice.get_range_original(Range::Normalized(0..3)),
            Some("ood"),
        );

        // If we keep only the modified piece
        let slice = s.slice(Range::Original(3..16)).unwrap();
        assert_eq!(
            slice.get_range_original(Range::Normalized(0..4)),
            Some("Good"),
        );
    }

    #[test]
    fn test_replace() {
        // Simple
        let s = NormalizedString::from(" Hello   friend ")
            .replace(' ', "_")
            .unwrap();
        assert_eq!(s.normalized, "_Hello___friend_");
        let s = NormalizedString::from("aaaab").replace('a', "b").unwrap();
        assert_eq!(s.normalized, "bbbbb");

        // Overlapping
        let s = NormalizedString::from("aaaab").replace("aaa", "b").unwrap();
        assert_eq!(s.normalized, "bab");

        // Regex
        let re = Regex::new(r"\s+").unwrap();
        let s = NormalizedString::from(" Hello   friend ")
            .replace(re, "_")
            .unwrap();
        assert_eq!(s.normalized, "_Hello_friend_");
    }

    #[test]
    fn test_split() {
        use SplitDelimiterBehavior::*;
        let s = NormalizedString::from("The-final--countdown");

        let test = |behavior: SplitDelimiterBehavior, result: Vec<&str>| {
            let splits = s.split('-', behavior).unwrap();
            assert_eq!(
                splits
                    .iter()
                    .map(|n| n.normalized.as_str())
                    .collect::<Vec<_>>(),
                result
            );
        };

        test(Removed, vec!["The", "final", "countdown"]);
        test(Isolated, vec!["The", "-", "final", "-", "-", "countdown"]);
        test(MergedWithPrevious, vec!["The-", "final-", "-", "countdown"]);
        test(MergedWithNext, vec!["The", "-final", "-", "-countdown"]);
        test(Contiguous, vec!["The", "-", "final", "--", "countdown"]);
    }

    #[test]
    fn test_transform_range_single_bytes() {
        let s = NormalizedString::from("Hello friend");

        // Removing at the beginning
        let current = s
            .clone()
            .transform_range(Range::Original(0..4), vec![('Y', 0)], 3);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );

        assert_eq!(
            current.alignments_original(),
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
        let current = s.clone().transform_range(
            Range::Original(3..10),
            vec![('_', 0), ('F', 0), ('R', -2)],
            2,
        );
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );

        assert_eq!(
            current.alignments_original(),
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
        let current = s
            .clone()
            .transform_range(Range::Original(5..), vec![('_', 0), ('F', -5)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        let current = s
            .clone()
            .transform_range(Range::Original(0..1), vec![('H', 1), ('H', 0)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        let current = s
            .clone()
            .transform_range(Range::Original(0..0), vec![('H', 1)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        let current = s
            .clone()
            .transform_range(Range::Original(0..1), vec![('H', 0), ('H', 1)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        let current = s.clone().transform_range(
            Range::Original(5..6),
            vec![('_', 0), ('m', 1), ('y', 1), ('_', 1)],
            0,
        );
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        let current =
            s.transform_range(Range::Original(11..), vec![('d', 0), ('_', 1), ('!', 1)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        let s = NormalizedString::from("𝔾𝕠𝕠𝕕");

        // Removing at the beginning
        let current = s
            .clone()
            .transform_range(Range::Original(0..8), vec![('G', -1)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        assert_eq!(current.get_range(Range::Original(0..8)).unwrap(), "G");
        assert_eq!(current.get_range(Range::Original(0..4)).unwrap(), "G");
        assert_eq!(
            current.get_range_original(Range::Original(0..4)).unwrap(),
            "𝔾",
        );
        assert_eq!(
            current.get_range_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠",
        );

        // Removing in the middle
        let current = s
            .clone()
            .transform_range(Range::Original(4..12), vec![('o', -1)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        let current = s
            .clone()
            .transform_range(Range::Original(12..), vec![('d', 0), ('!', 1)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );

        // Adding at the beginning
        let current = s
            .clone()
            .transform_range(Range::Original(0..4), vec![('_', 1), ('𝔾', 0)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        assert_eq!(current.get_range(Range::Original(0..8)).unwrap(), "𝔾𝕠");
        assert_eq!(current.get_range(Range::Original(0..4)).unwrap(), "𝔾");
        assert_eq!(
            current.get_range_original(Range::Original(0..4)).unwrap(),
            "𝔾",
        );
        assert_eq!(
            current.get_range_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠",
        );

        // Equivalent to the previous one
        let current = s
            .clone()
            .transform_range(Range::Original(0..0), vec![('_', 1)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        assert_eq!(current.get_range(Range::Original(0..8)).unwrap(), "𝔾𝕠");
        assert_eq!(current.get_range(Range::Original(0..4)).unwrap(), "𝔾");
        assert_eq!(
            current.get_range_original(Range::Original(0..4)).unwrap(),
            "𝔾",
        );
        assert_eq!(
            current.get_range_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠",
        );

        // Adding as part of the first character
        let current = s
            .clone()
            .transform_range(Range::Original(0..4), vec![('𝔾', 0), ('o', 1)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        assert_eq!(current.get_range(Range::Original(0..8)).unwrap(), "𝔾o𝕠");
        assert_eq!(current.get_range(Range::Original(0..4)).unwrap(), "𝔾o");
        assert_eq!(
            current.get_range_original(Range::Original(0..4)).unwrap(),
            "𝔾",
        );
        assert_eq!(
            current.get_range_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠",
        );

        // Adding in the middle
        let current = s.clone().transform_range(
            Range::Original(4..8),
            vec![('𝕠', 0), ('o', 1), ('o', 1), ('o', 1)],
            0,
        );
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        let current = s.transform_range(Range::Original(16..), vec![('!', 1)], 0);
        assert_eq!(
            current,
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
                original_shift: 0,
            },
        );
        assert_eq!(
            current.alignments_original(),
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
        let transforms = vec![('a', -2), ('…', 0)];
        let s = NormalizedString::from("abc…")
            .nfd()
            .transform(transforms, 0)
            .lowercase();
        assert_eq!(s.normalized, "a…");
    }

    macro_rules! test_pattern {
        ($inside: expr, $pattern: expr => @ERROR) => {
            assert!($pattern.find_matches($inside).is_err());
        };
        ($inside: expr, $pattern: expr => $result: expr) => {
            assert_eq!($pattern.find_matches($inside).unwrap(), $result);
            assert_eq!(
                Invert($pattern).find_matches($inside).unwrap(),
                $result
                    .into_iter()
                    .map(|v: (Offsets, bool)| (v.0, !v.1))
                    .collect::<Vec<_>>()
            );
        };
    }

    #[test]
    fn test_pattern_char() {
        test_pattern!(
            "aba", 'a' => vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), false),
                (Offsets(2, 3), true),
            ]
        );
        test_pattern!("bbbba", 'a' => vec![(Offsets(0, 4), false), (Offsets(4, 5), true)]);
        test_pattern!(
            "aabbb", 'a' => vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), true),
                (Offsets(2, 5), false),
            ]
        );
        test_pattern!("", 'a' => vec![(Offsets(0, 0), false)]);
        test_pattern!("aaa", 'b' => vec![(Offsets(0, 3), false)]);
    }

    #[test]
    fn test_pattern_str() {
        test_pattern!(
            "aba", "a" => vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), false),
                (Offsets(2, 3), true),
            ]
        );
        test_pattern!("bbbba", "a" => vec![(Offsets(0, 4), false), (Offsets(4, 5), true)]);
        test_pattern!(
            "aabbb", "a" => vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), true),
                (Offsets(2, 5), false),
            ]
        );
        test_pattern!(
            "aabbb", "ab" => vec![
                (Offsets(0, 1), false),
                (Offsets(1, 3), true),
                (Offsets(3, 5), false),
            ]
        );
        test_pattern!(
            "aabbab", "ab" => vec![
                (Offsets(0, 1), false),
                (Offsets(1, 3), true),
                (Offsets(3, 4), false),
                (Offsets(4, 6), true),
            ]
        );
        test_pattern!("", "" => vec![(Offsets(0, 0), false)]);
        test_pattern!("aaa", "" => vec![(Offsets(0, 3), false)]);
        test_pattern!("aaa", "b" => vec![(Offsets(0, 3), false)]);
    }

    #[test]
    fn test_pattern_functions() {
        let is_b = |c| c == 'b';
        test_pattern!(
            "aba", is_b => vec![
                (Offsets(0, 1), false),
                (Offsets(1, 2), true),
                (Offsets(2, 3), false),
            ]
        );
        test_pattern!("aaaab", is_b => vec![(Offsets(0, 4), false), (Offsets(4, 5), true)]);
        test_pattern!(
            "bbaaa", is_b => vec![
                (Offsets(0, 1), true),
                (Offsets(1, 2), true),
                (Offsets(2, 5), false),
            ]
        );
        test_pattern!("", is_b => vec![(Offsets(0, 0), false)]);
        test_pattern!("aaa", is_b => vec![(Offsets(0, 3), false)]);
    }

    #[test]
    fn test_pattern_regex() {
        let is_whitespace = Regex::new(r"\s+").unwrap();
        test_pattern!(
            "a   b", &is_whitespace => vec![
                (Offsets(0, 1), false),
                (Offsets(1, 4), true),
                (Offsets(4, 5), false),
            ]
        );
        test_pattern!(
            "   a   b   ", &is_whitespace => vec![
                (Offsets(0, 3), true),
                (Offsets(3, 4), false),
                (Offsets(4, 7), true),
                (Offsets(7, 8), false),
                (Offsets(8, 11), true),
            ]
        );
        test_pattern!("", &is_whitespace => vec![(Offsets(0, 0), false)]);
        test_pattern!(
            "𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘", &is_whitespace => vec![
                (Offsets(0, 16), false),
                (Offsets(16, 17), true),
                (Offsets(17, 45), false),
            ]
        );
        test_pattern!("aaa", &is_whitespace => vec![(Offsets(0, 3), false)]);
    }
}
