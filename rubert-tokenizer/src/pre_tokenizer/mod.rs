pub mod string;

use unicode_categories::UnicodeCategories;

use crate::{
    normalizer::string::{NormalizedString, SplitDelimiterBehavior},
    pre_tokenizer::string::PreTokenizedString,
    Error,
};

/// A Bert pre-tokenizer.
pub struct PreTokenizer;

impl PreTokenizer {
    pub(crate) fn pre_tokenize(
        &self,
        normalized: NormalizedString,
    ) -> Result<PreTokenizedString, Error> {
        PreTokenizedString::from(normalized)
            .split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))?
            .split(|_, s| {
                s.split(
                    |c: char| c.is_ascii_punctuation() || c.is_punctuation(),
                    SplitDelimiterBehavior::Isolated,
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizer::string::Offsets;

    fn assert_eq(actual: PreTokenizedString, expected: Vec<(&str, Offsets)>) {
        assert_eq!(actual.splits.len(), expected.len());
        for (split, (string, offset)) in actual.splits.iter().zip(expected) {
            assert_eq!(split.normalized, string);
            assert_eq!(
                split.original_shift + split.alignments.first().unwrap().0,
                offset.0,
            );
            assert_eq!(
                split.original_shift + split.alignments.last().unwrap().1,
                offset.1,
            );
        }
    }

    #[test]
    fn basic() {
        let normalized = "Hey friend!     How are you?!?".into();
        let pre_tokenized = PreTokenizer.pre_tokenize(normalized).unwrap();
        let expected = vec![
            ("Hey", Offsets(0, 3)),
            ("friend", Offsets(4, 10)),
            ("!", Offsets(10, 11)),
            ("How", Offsets(16, 19)),
            ("are", Offsets(20, 23)),
            ("you", Offsets(24, 27)),
            ("?", Offsets(27, 28)),
            ("!", Offsets(28, 29)),
            ("?", Offsets(29, 30)),
        ];
        assert_eq(pre_tokenized, expected);
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
        let pre_tokenized = PreTokenizer.pre_tokenize(normalized).unwrap();
        let expected = vec![
            ("野", Offsets(0, 3)),
            ("口", Offsets(3, 6)),
            ("里", Offsets(6, 9)),
            ("佳", Offsets(9, 12)),
            ("Noguchi", Offsets(13, 20)),
            ("Rika", Offsets(21, 25)),
        ];
        assert_eq(pre_tokenized, expected);
    }
}
