use std::{borrow::Borrow, collections::HashMap};

use derive_more::{Deref, From};
use ndarray::Array2;

use crate::tokenizer::encoding::ActiveMask;

/// The collection of all potential key phrases.
pub struct KeyPhrases {
    choices: Vec<String>,
    mentions: Vec<i64>,
}

/// The ranked key phrases in descending order.
#[derive(Clone, Debug, Deref, From)]
pub struct RankedKeyPhrases(Vec<String>);

impl KeyPhrases {
    /// Collects all potential key phrases from the words.
    ///
    /// Each key phrase contains at most `size` words.
    pub fn collect(words: &[impl Borrow<str>], size: usize) -> Self {
        if words.is_empty() {
            return Self {
                choices: Vec::new(),
                mentions: Vec::new(),
            };
        }

        let num_words = words.len();
        let size = size.min(num_words);
        let max_key_phrases = (0..size).into_iter().fold(0, |max, n| max + num_words - n);

        let mut choices = Vec::with_capacity(max_key_phrases);
        let mut mentions = Vec::with_capacity(max_key_phrases);
        for n in 0..size {
            let max_key_phrases = num_words - n;
            let mut key_phrase_idx = HashMap::with_capacity(max_key_phrases);
            for i in 0..max_key_phrases {
                let key_phrase = words[i..i + n + 1].join(" ");
                let idx = *key_phrase_idx.entry(key_phrase.clone()).or_insert_with(|| {
                    choices.push(key_phrase);
                    choices.len() as i64 - 1
                });
                mentions.push(idx);
            }
        }
        choices.shrink_to_fit();

        debug_assert_eq!(choices.len() as i64 - 1, *mentions.iter().max().unwrap());
        debug_assert_eq!(mentions.len(), max_key_phrases);
        debug_assert!(choices.len() <= mentions.len());

        Self { choices, mentions }
    }

    /// Creates the mask of active/mentioned key phrases for each unique keyphrase.
    pub fn active_mask(&self) -> ActiveMask {
        Array2::from_shape_fn((self.choices.len(), self.mentions.len()), |(i, j)| {
            i as i64 == self.mentions[j]
        })
        .into()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayView2;

    use super::*;

    const UNIQUE_WORDS: [&str; 4] = ["this", "embedding", "fits", "perfectly"];
    const DUPLICATE_WORDS: [&str; 9] = [
        "this",
        "embedding",
        "fits",
        "perfectly",
        "and",
        "this",
        "embedding",
        "fits",
        "well",
    ];
    const FEW_WORDS: [&str; 2] = ["this", "embedding"];

    #[test]
    fn test_collect_unique() {
        let size = 3;
        let key_phrases = KeyPhrases::collect(&UNIQUE_WORDS, size);
        assert_eq!(
            key_phrases.choices,
            [
                "this",
                "embedding",
                "fits",
                "perfectly",
                "this embedding",
                "embedding fits",
                "fits perfectly",
                "this embedding fits",
                "embedding fits perfectly",
            ],
        );
        assert_eq!(
            key_phrases.mentions,
            [
                0, 1, 2, 3, //
                4, 5, 6, //
                7, 8, //
            ],
        );
    }

    #[test]
    fn test_collect_duplicate() {
        let size = 3;
        let key_phrases = KeyPhrases::collect(&DUPLICATE_WORDS, size);
        assert_eq!(
            key_phrases.choices,
            [
                "this",
                "embedding",
                "fits",
                "perfectly",
                "and",
                "well",
                "this embedding",
                "embedding fits",
                "fits perfectly",
                "perfectly and",
                "and this",
                "fits well",
                "this embedding fits",
                "embedding fits perfectly",
                "fits perfectly and",
                "perfectly and this",
                "and this embedding",
                "embedding fits well",
            ],
        );
        assert_eq!(
            key_phrases.mentions,
            [
                0, 1, 2, 3, 4, 0, 1, 2, 5, //
                6, 7, 8, 9, 10, 6, 7, 11, //
                12, 13, 14, 15, 16, 12, 17, //
            ],
        );
    }

    #[test]
    fn test_collect_few() {
        let size = 3;
        let key_phrases = KeyPhrases::collect(&FEW_WORDS, size);
        assert_eq!(key_phrases.choices, ["this", "embedding", "this embedding"]);
        assert_eq!(
            key_phrases.mentions,
            [
                0, 1, //
                2, //
            ],
        );
    }

    #[test]
    fn test_collect_empty() {
        let key_phrases = KeyPhrases::collect(&[] as &[&str], 3);
        assert!(key_phrases.choices.is_empty());
        assert!(key_phrases.mentions.is_empty());
    }

    #[test]
    fn test_mask_unique() {
        assert_eq!(
            KeyPhrases::collect(&UNIQUE_WORDS, 3).active_mask().0,
            ArrayView2::from_shape(
                (9, 9),
                &[
                    true, false, false, false, false, false, false, false, false, //
                    false, true, false, false, false, false, false, false, false, //
                    false, false, true, false, false, false, false, false, false, //
                    false, false, false, true, false, false, false, false, false, //
                    false, false, false, false, true, false, false, false, false, //
                    false, false, false, false, false, true, false, false, false, //
                    false, false, false, false, false, false, true, false, false, //
                    false, false, false, false, false, false, false, true, false, //
                    false, false, false, false, false, false, false, false, true, //
                ],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_mask_duplicate() {
        assert_eq!(
            KeyPhrases::collect(&DUPLICATE_WORDS, 3).active_mask().0,
            ArrayView2::from_shape(
                (18, 24),
                &[
                    true, false, false, false, false, true, false, false, false, false, false,
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, //
                    false, true, false, false, false, false, true, false, false, false, false,
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, //
                    false, false, true, false, false, false, false, true, false, false, false,
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, //
                    false, false, false, true, false, false, false, false, false, false, false,
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, //
                    false, false, false, false, true, false, false, false, false, false, false,
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, true, false, false,
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, true, false,
                    false, false, false, true, false, false, false, false, false, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, true,
                    false, false, false, false, true, false, false, false, false, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    true, false, false, false, false, false, false, false, false, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, true, false, false, false, false, false, false, false, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, true, false, false, false, false, false, false, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, false, false, false, true, false, false, false, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, false, false, false, false, true, false, false, false, false,
                    true, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, false, false, false, false, false, true, false, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, false, false, false, false, false, false, true, false, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, false, false, false, false, false, false, false, true, false,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, false, false, false, false, false, false, false, false, true,
                    false, false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, false, false, false, false, false, false, false, false, false, false,
                    false, true, //
                ],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_mask_few() {
        assert_eq!(
            KeyPhrases::collect(&FEW_WORDS, 3).active_mask().0,
            ArrayView2::from_shape(
                (3, 3),
                &[
                    true, false, false, //
                    false, true, false, //
                    false, false, true, //
                ],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_mask_empty() {
        assert_eq!(
            KeyPhrases::collect(&[] as &[&str], 3).active_mask().0,
            ArrayView2::from_shape((0, 0), &[] as &[bool]).unwrap(),
        );
    }
}
