use derive_more::{Deref, From};
use ndarray::{Array1, Array2, Axis};

use crate::tokenizer::{key_phrase::KeyPhrases, Tokenizer};

/// The token ids of the encoded sequence.
///
/// The token ids are of shape `(1, token_size)`.
#[derive(Clone, Deref, From)]
pub struct TokenIds(pub Array2<i64>);

/// The attention mask of the encoded sequence.
///
/// The attention mask is of shape `(1, token_size)`.
#[derive(Clone, Deref, From)]
pub struct AttentionMask(pub Array2<i64>);

/// The type ids of the encoded sequence.
///
/// The type ids are of shape `(1, token_size)`.
#[derive(Clone, Deref, From)]
pub struct TypeIds(pub Array2<i64>);

/// The starting tokens mask of the encoded sequence.
///
/// The valid mask is of shape `(token_size,)`.
#[derive(Clone, Deref, From)]
pub struct ValidMask(pub Vec<bool>);

/// The active words mask for each key phrase.
///
/// The active mask is of shape `(key_phrase_choices, key_phrase_mentions)`.
#[derive(Clone, Deref, From)]
pub struct ActiveMask(pub Array2<bool>);

/// The encoded sequence.
pub struct Encoding {
    pub token_ids: TokenIds,
    pub attention_mask: AttentionMask,
    pub type_ids: TypeIds,
    pub valid_mask: ValidMask,
    pub active_mask: ActiveMask,
}

impl<const KEY_PHRASE_SIZE: usize> Tokenizer<KEY_PHRASE_SIZE> {
    /// Encodes the sequence.
    ///
    /// The encoding is in correct shape for the models.
    pub fn encode(&self, sequence: impl AsRef<str>) -> (Encoding, KeyPhrases<KEY_PHRASE_SIZE>) {
        let encoding = self.tokenizer.encode(sequence);
        let (token_ids, type_ids, tokens, word_indices, _, _, attention_mask, overflowing) =
            encoding.into();

        let token_ids = Array1::from(token_ids).insert_axis(Axis(0)).into();
        let attention_mask = Array1::from(attention_mask).insert_axis(Axis(0)).into();
        let type_ids = Array1::from(type_ids).insert_axis(Axis(0)).into();

        let valid_mask = valid_mask(&word_indices);
        let words = decode_words(tokens, word_indices, overflowing);
        let key_phrases =
            KeyPhrases::collect(&words, self.key_phrase_max_count, self.key_phrase_min_score);
        let active_mask = key_phrases.active_mask();

        (
            Encoding {
                token_ids,
                attention_mask,
                type_ids,
                valid_mask,
                active_mask,
            },
            key_phrases,
        )
    }
}

/// Joins starting tokens with their continuing tokens to decode the tokenized words.
fn decode_words(
    tokens: Vec<String>,
    word_indices: Vec<Option<i64>>,
    overflowing: Option<Vec<rubert_tokenizer::Encoding<i64>>>,
) -> Vec<String> {
    let mut words = Vec::<String>::with_capacity(word_indices.len());
    let last_idx = word_indices.into_iter().zip(tokens.into_iter()).fold(
        None,
        |previous, (current, token)| {
            if current.is_some() {
                if previous == current {
                    words.last_mut().unwrap().push_str(&token[2..]);
                } else {
                    words.push(token);
                }
                current
            } else {
                previous
            }
        },
    );
    // subtokens of the last word might have been truncated during tokenization, but we can
    // still use the whole word for the keyphrase because the model only pays attention to the
    // starting token
    if let Some(overflowing) = overflowing {
        if !overflowing.is_empty() {
            for (idx, token) in overflowing[0]
                .word_indices()
                .iter()
                .zip(overflowing[0].tokens().iter())
            {
                if idx.is_some() {
                    if idx == &last_idx {
                        words.last_mut().unwrap().push_str(&token[2..]);
                    } else {
                        break;
                    }
                }
            }
        }
    }
    words.shrink_to_fit();

    words
}

/// Creates the mask of starting tokens.
fn valid_mask(word_indices: &[Option<i64>]) -> ValidMask {
    word_indices
        .iter()
        .scan(None, |previous, current| {
            if current == previous {
                Some(false)
            } else {
                *previous = *current;
                if current.is_some() {
                    Some(true)
                } else {
                    Some(false)
                }
            }
        })
        .collect::<Vec<_>>()
        .into()
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use ndarray::ArrayView2;

    use super::*;
    use test_utils::smbert::vocab;

    const EXACT_SEQUENCE: &str = "This embedding fits perfectly.";
    const SHORT_SEQUENCE: &str = "This is an embedding.";
    const LONG_SEQUENCE: &str = "This embedding is way too long.";

    fn tokenizer(token_size: usize) -> Tokenizer<3> {
        let vocab = BufReader::new(File::open(vocab().unwrap()).unwrap());
        let accents = false;
        let lowercase = true;
        let key_phrase_count = None;
        let key_phrase_score = None;

        Tokenizer::new(
            vocab,
            accents,
            lowercase,
            token_size,
            key_phrase_count,
            key_phrase_score,
        )
        .unwrap()
    }

    #[test]
    fn test_encode_exact() {
        let shape = (1, 10);
        let tokenizer = tokenizer(shape.1);
        let (encoding, _) = tokenizer.encode(EXACT_SEQUENCE);
        assert_eq!(
            encoding.token_ids.0,
            ArrayView2::from_shape(
                shape,
                &[2, 2584, 69469, 1599, 13891, 1046, 18992, 1838, 5, 3],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_mask.0,
            ArrayView2::from_shape(shape, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap(),
        );
        assert_eq!(
            encoding.type_ids.0,
            ArrayView2::from_shape(shape, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
        assert_eq!(
            encoding.valid_mask.0,
            [false, true, true, false, true, false, true, false, true, false],
        );
        assert_eq!(
            encoding.active_mask.0,
            ArrayView2::from_shape(
                (12, 12),
                &[
                    true, false, false, false, false, false, false, false, false, false, false,
                    false, //
                    false, true, false, false, false, false, false, false, false, false, false,
                    false, //
                    false, false, true, false, false, false, false, false, false, false, false,
                    false, //
                    false, false, false, true, false, false, false, false, false, false, false,
                    false, //
                    false, false, false, false, true, false, false, false, false, false, false,
                    false, //
                    false, false, false, false, false, true, false, false, false, false, false,
                    false, //
                    false, false, false, false, false, false, true, false, false, false, false,
                    false, //
                    false, false, false, false, false, false, false, true, false, false, false,
                    false, //
                    false, false, false, false, false, false, false, false, true, false, false,
                    false, //
                    false, false, false, false, false, false, false, false, false, true, false,
                    false, //
                    false, false, false, false, false, false, false, false, false, false, true,
                    false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    true, //
                ],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_encode_padded() {
        let shape = (1, 10);
        let tokenizer = tokenizer(shape.1);
        let (encoding, _) = tokenizer.encode(SHORT_SEQUENCE);
        assert_eq!(
            encoding.token_ids.0,
            ArrayView2::from_shape(shape, &[2, 2584, 1693, 1624, 69469, 1599, 5, 3, 0, 0]).unwrap(),
        );
        assert_eq!(
            encoding.attention_mask.0,
            ArrayView2::from_shape(shape, &[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]).unwrap(),
        );
        assert_eq!(
            encoding.type_ids.0,
            ArrayView2::from_shape(shape, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
        assert_eq!(
            encoding.valid_mask.0,
            [false, true, true, true, true, false, true, false, false, false],
        );
        assert_eq!(
            encoding.active_mask.0,
            ArrayView2::from_shape(
                (12, 12),
                &[
                    true, false, false, false, false, false, false, false, false, false, false,
                    false, //
                    false, true, false, false, false, false, false, false, false, false, false,
                    false, //
                    false, false, true, false, false, false, false, false, false, false, false,
                    false, //
                    false, false, false, true, false, false, false, false, false, false, false,
                    false, //
                    false, false, false, false, true, false, false, false, false, false, false,
                    false, //
                    false, false, false, false, false, true, false, false, false, false, false,
                    false, //
                    false, false, false, false, false, false, true, false, false, false, false,
                    false, //
                    false, false, false, false, false, false, false, true, false, false, false,
                    false, //
                    false, false, false, false, false, false, false, false, true, false, false,
                    false, //
                    false, false, false, false, false, false, false, false, false, true, false,
                    false, //
                    false, false, false, false, false, false, false, false, false, false, true,
                    false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    true, //
                ],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_encode_truncated() {
        let shape = (1, 8);
        let tokenizer = tokenizer(shape.1);
        let (encoding, _) = tokenizer.encode(LONG_SEQUENCE);
        assert_eq!(
            encoding.token_ids.0,
            ArrayView2::from_shape(shape, &[2, 2584, 69469, 1599, 1693, 5331, 11700, 3]).unwrap(),
        );
        assert_eq!(
            encoding.attention_mask.0,
            ArrayView2::from_shape(shape, &[1, 1, 1, 1, 1, 1, 1, 1]).unwrap(),
        );
        assert_eq!(
            encoding.type_ids.0,
            ArrayView2::from_shape(shape, &[0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
        assert_eq!(
            encoding.valid_mask.0,
            [false, true, true, false, true, true, true, false],
        );
        assert_eq!(
            encoding.active_mask.0,
            ArrayView2::from_shape(
                (12, 12),
                &[
                    true, false, false, false, false, false, false, false, false, false, false,
                    false, //
                    false, true, false, false, false, false, false, false, false, false, false,
                    false, //
                    false, false, true, false, false, false, false, false, false, false, false,
                    false, //
                    false, false, false, true, false, false, false, false, false, false, false,
                    false, //
                    false, false, false, false, true, false, false, false, false, false, false,
                    false, //
                    false, false, false, false, false, true, false, false, false, false, false,
                    false, //
                    false, false, false, false, false, false, true, false, false, false, false,
                    false, //
                    false, false, false, false, false, false, false, true, false, false, false,
                    false, //
                    false, false, false, false, false, false, false, false, true, false, false,
                    false, //
                    false, false, false, false, false, false, false, false, false, true, false,
                    false, //
                    false, false, false, false, false, false, false, false, false, false, true,
                    false, //
                    false, false, false, false, false, false, false, false, false, false, false,
                    true, //
                ],
            )
            .unwrap(),
        );
    }

    const EXACT_WORDS: [&str; 5] = ["this", "embedding", "fits", "perfectly", "."];
    const SHORT_WORDS: [&str; 5] = ["this", "is", "an", "embedding", "."];
    const LONG_WORDS: [&str; 7] = ["this", "embedding", "is", "way", "too", "long", "."];

    #[test]
    fn test_decode_words_exact() {
        let (_, _, tokens, word_indices, _, _, _, overflowing) =
            tokenizer(10).tokenizer.encode(EXACT_SEQUENCE).into();
        let words = decode_words(tokens, word_indices, overflowing);
        assert_eq!(words, EXACT_WORDS);
    }

    #[test]
    fn test_decode_words_padded() {
        let (_, _, tokens, word_indices, _, _, _, overflowing) =
            tokenizer(10).tokenizer.encode(SHORT_SEQUENCE).into();
        let words = decode_words(tokens, word_indices, overflowing);
        assert_eq!(words, SHORT_WORDS);
    }

    #[test]
    fn test_decode_words_truncated_between() {
        let (_, _, tokens, word_indices, _, _, _, overflowing) =
            tokenizer(8).tokenizer.encode(LONG_SEQUENCE).into();
        let words = decode_words(tokens, word_indices, overflowing);
        assert_eq!(words, LONG_WORDS[..5]);
    }

    #[test]
    fn test_decode_words_truncated_within() {
        let (_, _, tokens, word_indices, _, _, _, overflowing) =
            tokenizer(4).tokenizer.encode(LONG_SEQUENCE).into();
        let words = decode_words(tokens, word_indices, overflowing);
        assert_eq!(words, LONG_WORDS[..2]);
    }

    #[test]
    fn test_decode_words_truncated_empty() {
        let (_, _, tokens, word_indices, _, _, _, overflowing) =
            tokenizer(2).tokenizer.encode(LONG_SEQUENCE).into();
        let words = decode_words(tokens, word_indices, overflowing);
        assert!(words.is_empty());
    }

    #[test]
    fn test_decode_words_empty() {
        let (_, _, tokens, word_indices, _, _, _, overflowing) =
            tokenizer(5).tokenizer.encode("").into();
        let words = decode_words(tokens, word_indices, overflowing);
        assert!(words.is_empty());
    }

    #[test]
    fn test_valid_mask_full() {
        let word_indices = vec![
            None,
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            Some(3),
            Some(3),
            Some(4),
            None,
        ];
        assert_eq!(
            valid_mask(&word_indices).0,
            [false, true, true, false, true, true, false, true, false],
        );
    }

    #[test]
    fn test_valid_mask_empty() {
        assert_eq!(valid_mask(&[]).0, [] as [bool; 0]);
    }
}
