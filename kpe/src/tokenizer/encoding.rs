use std::ops::ControlFlow;

use derive_more::{Deref, From};
use ndarray::{Array1, Array2, Axis};

use crate::tokenizer::{key_phrase::KeyPhrases, Tokenizer};
use rubert_tokenizer::{Encoding as BertEncoding, Offsets};

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
        let sequence = sequence.as_ref();
        let encoding = self.tokenizer.encode(sequence);
        let (token_ids, type_ids, _, _, ref offsets, _, attention_mask, overflowing) =
            encoding.into();

        let token_ids = Array1::from(token_ids).insert_axis(Axis(0)).into();
        let attention_mask = Array1::from(attention_mask).insert_axis(Axis(0)).into();
        let type_ids = Array1::from(type_ids).insert_axis(Axis(0)).into();

        let valid_mask = valid_mask(offsets);
        let words = decode_words(sequence, offsets, overflowing);
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

/// Decodes the tokenized words.
///
/// Joins starting tokens with their continuing tokens. Everything which is not separated by
/// whitespace is considered as continuation as well, e.g. punctuation.
fn decode_words(
    sequence: impl AsRef<str>,
    offsets: impl AsRef<[Offsets]>,
    overflowing: Option<impl AsRef<[BertEncoding<i64>]>>,
) -> Vec<String> {
    let sequence = sequence.as_ref();
    let offsets = offsets.as_ref();
    let mut words = Vec::<String>::with_capacity(offsets.len());
    let (word_start, word_end) = offsets.iter().fold(
        (0, 0),
        |(word_start, word_end), &Offsets(token_start, token_end)| {
            if word_end < token_start {
                words.push(sequence[word_start..word_end].into());
                (token_start, token_end)
            } else {
                (word_start, token_end)
            }
        },
    );
    if word_start < word_end {
        if let Some(overflowing) = overflowing {
            // subtokens of the last word might have been truncated during tokenization, but we can
            // still use the whole word for the keyphrase because the model only pays attention to
            // the starting token
            if let ControlFlow::Continue((word_start, word_end)) = overflowing
                .as_ref()
                .iter()
                .flat_map(BertEncoding::offsets)
                .try_fold(
                    (word_start, word_end),
                    |(word_start, word_end), &Offsets(token_start, token_end)| {
                        if word_end < token_start {
                            words.push(sequence[word_start..word_end].into());
                            ControlFlow::Break(())
                        } else {
                            ControlFlow::Continue((word_start, token_end))
                        }
                    },
                )
            {
                words.push(sequence[word_start..word_end].into());
            }
        } else {
            words.push(sequence[word_start..word_end].into());
        }
    }
    words.shrink_to_fit();

    words
}

/// Creates the mask of starting tokens.
fn valid_mask(offsets: impl AsRef<[Offsets]>) -> ValidMask {
    offsets
        .as_ref()
        .iter()
        .scan(0, |previous_end, &Offsets(start, end)| {
            let valid = start > *previous_end || (start == 0 && end > 0);
            *previous_end = end;
            Some(valid)
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

    /// Tokens: This embedd ##ing fit ##s perfect ##ly .
    const EXACT_SEQUENCE: &str = "This embedding fits perfectly.";
    /// Tokens: This is an embedd ##ing .
    const SHORT_SEQUENCE: &str = "This is an embedding.";
    /// Tokens: This embedd ##ing is way too long .
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
            [false, true, true, false, true, false, true, false, false, false],
        );
        assert_eq!(
            encoding.active_mask.0,
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
            [false, true, true, true, true, false, false, false, false, false],
        );
        assert_eq!(
            encoding.active_mask.0,
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

    const EXACT_WORDS: [&str; 4] = ["This", "embedding", "fits", "perfectly."];
    const SHORT_WORDS: [&str; 4] = ["This", "is", "an", "embedding."];
    const LONG_WORDS: [&str; 6] = ["This", "embedding", "is", "way", "too", "long."];

    #[test]
    fn test_decode_words_exact() {
        let encoding = tokenizer(10).tokenizer.encode(EXACT_SEQUENCE);
        let words = decode_words(EXACT_SEQUENCE, encoding.offsets(), encoding.overflowing());
        assert_eq!(words, EXACT_WORDS);
    }

    #[test]
    fn test_decode_words_padded() {
        let encoding = tokenizer(10).tokenizer.encode(SHORT_SEQUENCE);
        let words = decode_words(SHORT_SEQUENCE, encoding.offsets(), encoding.overflowing());
        assert_eq!(words, SHORT_WORDS);
    }

    #[test]
    fn test_decode_words_truncated_between() {
        let encoding = tokenizer(8).tokenizer.encode(LONG_SEQUENCE);
        let words = decode_words(LONG_SEQUENCE, encoding.offsets(), encoding.overflowing());
        assert_eq!(words, LONG_WORDS[..5]);
    }

    #[test]
    fn test_decode_words_truncated_within() {
        let encoding = tokenizer(4).tokenizer.encode(LONG_SEQUENCE);
        let words = decode_words(LONG_SEQUENCE, encoding.offsets(), encoding.overflowing());
        assert_eq!(words, LONG_WORDS[..2]);
    }

    #[test]
    fn test_decode_words_truncated_empty() {
        let encoding = tokenizer(2).tokenizer.encode(LONG_SEQUENCE);
        let words = decode_words(LONG_SEQUENCE, encoding.offsets(), encoding.overflowing());
        assert!(words.is_empty());
    }

    #[test]
    fn test_decode_words_empty() {
        let encoding = tokenizer(5).tokenizer.encode("");
        let words = decode_words("", encoding.offsets(), encoding.overflowing());
        assert!(words.is_empty());
    }

    #[test]
    fn test_valid_mask_full() {
        let encoding = tokenizer(10).tokenizer.encode(EXACT_SEQUENCE);
        assert_eq!(
            valid_mask(encoding.offsets()).0,
            [false, true, true, false, true, false, true, false, false, false],
        );
    }

    #[test]
    fn test_valid_mask_empty() {
        assert!(valid_mask(&[]).0.is_empty());
    }
}
