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
    use crate::{
        normalizer::string::{OffsetReferential, Offsets},
        pre_tokenizer::string::OffsetType,
    };

    #[test]
    fn basic() {
        let normalized = "Hey friend!     How are you?!?".into();
        let pre_tokenized = PreTokenizer.pre_tokenize(normalized).unwrap();
        assert_eq!(
            pre_tokenized.get_splits(OffsetReferential::Original, OffsetType::Byte),
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
        let pretokenized = PreTokenizer.pre_tokenize(normalized).unwrap();
        assert_eq!(
            pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte),
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
