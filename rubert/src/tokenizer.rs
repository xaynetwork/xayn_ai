use std::io::BufRead;

use derive_more::{Deref, From};
use displaydoc::Display;
use rubert_tokenizer::{Builder, BuilderError, Padding, Tokenizer as BertTokenizer, Truncation};
use thiserror::Error;

use crate::ndarray::{Array2, Dim, Ix2};

/// A [`RuBert`] tokenizer.
///
/// Wraps a pre-configured Bert tokenizer.
///
/// [`RuBert`]: crate::pipeline::RuBert
pub struct Tokenizer {
    tokenizer: BertTokenizer<i64>,
    input_shape: Ix2,
}

/// Potential errors of the [`RuBert`] [`Tokenizer`].
///
/// [`RuBert`]: crate::pipeline::RuBert
#[derive(Debug, Display, Error)]
pub enum TokenizerError {
    /// Failed to build the tokenizer: {0}
    Builder(#[from] BuilderError),
}

/// The input ids of the encoded sentences.
#[derive(Clone, Deref, From)]
pub struct InputIds(pub(crate) Array2<i64>);

/// The attention masks of the encoded sentences.
#[derive(Clone, Deref, From)]
pub struct AttentionMasks(pub(crate) Array2<i64>);

/// The token type ids of the encoded sentences.
#[derive(Clone, Deref, From)]
pub struct TokenTypeIds(pub(crate) Array2<i64>);

/// The encoded sentences.
pub struct Encodings {
    pub(crate) input_ids: InputIds,
    pub(crate) attention_masks: AttentionMasks,
    pub(crate) token_type_ids: TokenTypeIds,
}

impl Tokenizer {
    /// Creates a [`RuBert`] tokenizer from a vocabulary.
    ///
    /// Can be set to strip accents and to lowercase the sequences. Requires the maximum number of
    /// sequences as well as tokens per tokenized sequence, which applies to padding and truncation
    /// and includes special tokens as well.
    ///
    /// [`RuBert`]: crate::pipeline::RuBert
    pub fn new(
        // `BufRead` instead of `AsRef<Path>` is needed for wasm
        vocab: impl BufRead,
        accents: bool,
        lowercase: bool,
        batch_size: usize,
        token_size: usize,
    ) -> Result<Self, TokenizerError> {
        let tokenizer = Builder::new(vocab)?
            .with_normalizer(true, false, accents, lowercase)
            .with_model("[UNK]", "##", 100)
            .with_post_tokenizer("[CLS]", "[SEP]")
            .with_truncation(Truncation::fixed(token_size, 0))
            .with_padding(Padding::fixed(token_size, "[PAD]"))
            .build()?;
        let input_shape = Dim([batch_size, token_size]);

        Ok(Tokenizer {
            tokenizer,
            input_shape,
        })
    }

    /// Encodes the sequences.
    ///
    /// The encodings are in correct shape for the model.
    pub fn encode(&self, sequences: &[impl AsRef<str>]) -> Encodings {
        let encodings = self.tokenizer.encode_batch(sequences);

        let input_ids = InputIds(Array2::from_shape_fn(self.input_shape, |(i, j)| {
            encodings
                .get(i)
                .map(|encoding| encoding.ids().get(j))
                .flatten()
                .copied()
                .unwrap_or(0)
        }));
        let attention_masks = AttentionMasks(Array2::from_shape_fn(self.input_shape, |(i, j)| {
            encodings
                .get(i)
                .map(|encoding| encoding.attention_mask().get(j))
                .flatten()
                .copied()
                .unwrap_or(0)
        }));
        let token_type_ids = TokenTypeIds(Array2::from_shape_fn(self.input_shape, |(i, j)| {
            encodings
                .get(i)
                .map(|encoding| encoding.type_ids().get(j))
                .flatten()
                .copied()
                .unwrap_or(0)
        }));

        Encodings {
            input_ids,
            attention_masks,
            token_type_ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use super::*;
    use crate::{ndarray::ArrayView, VOCAB};

    fn tokenizer(shape: (usize, usize)) -> Tokenizer {
        let vocab = BufReader::new(File::open(VOCAB).unwrap());
        let accents = true;
        let lowercase = true;
        Tokenizer::new(vocab, accents, lowercase, shape.0, shape.1).unwrap()
    }

    #[test]
    fn test_encode() {
        // too short
        let shape = (1, 20);
        let encoding = tokenizer(shape).encode(&["These are normal, common EMBEDDINGS."]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                shape,
                &[2, 4538, 2128, 8561, 1, 6541, 69469, 2762, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(
                shape,
                &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape(
                shape,
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );

        // too long
        let shape = (1, 10);
        let encoding = tokenizer(shape).encode(&["These are normal, common EMBEDDINGS."]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(shape, &[2, 4538, 2128, 8561, 1, 6541, 69469, 2762, 5, 3])
                .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(shape, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape(shape, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }

    #[test]
    fn test_encode_batch() {
        // both too short
        let shape = (2, 10);
        let encoding = tokenizer(shape).encode(&["a b c", "a b c d"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    2, 7, 8, 9, 3, 0, 0, 0, 0, 0, //
                    2, 7, 8, 9, 10, 3, 0, 0, 0, 0,
                ],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(
                shape,
                &[
                    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, //
                    1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                ]
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
            )
            .unwrap(),
        );

        // one too short and one too long
        let encoding = tokenizer(shape).encode(&["a b c", "a b c d e f g h i j k l"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    2, 7, 8, 9, 3, 0, 0, 0, 0, 0, //
                    2, 7, 8, 9, 10, 11, 12, 13, 14, 3
                ],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(
                shape,
                &[
                    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, //
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                ]
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
            )
            .unwrap(),
        );

        // both too long
        let encoding =
            tokenizer(shape).encode(&["a b c d e f g h i j k l", "a b c d e f g h i j k l m n"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    2, 7, 8, 9, 10, 11, 12, 13, 14, 3, //
                    2, 7, 8, 9, 10, 11, 12, 13, 14, 3,
                ],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(
                shape,
                &[
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                ]
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_encode_batch_edge_cases() {
        // first one too short, second fits
        let shape = (2, 5);
        let encoding = tokenizer(shape).encode(&["a b", "a b c"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    2, 7, 8, 3, 0, //
                    2, 7, 8, 9, 3,
                ],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(
                shape,
                &[
                    1, 1, 1, 1, 0, //
                    1, 1, 1, 1, 1,
                ]
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    0, 0, 0, 0, 0, //
                    0, 0, 0, 0, 0,
                ]
            )
            .unwrap(),
        );

        // first fits, second 1 too long
        let encoding = tokenizer(shape).encode(&["a b c", "a b c d"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    2, 7, 8, 9, 3, //
                    2, 7, 8, 9, 3
                ],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(
                shape,
                &[
                    1, 1, 1, 1, 1, //
                    1, 1, 1, 1, 1,
                ]
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape(
                shape,
                &[
                    0, 0, 0, 0, 0, //
                    0, 0, 0, 0, 0,
                ]
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_encode_troublemakers() {
        // known troublemakers
        let shape = (1, 15);
        let encoding = tokenizer(shape).encode(&["for “life-threatening storm surge” according"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                shape,
                &[2, 1665, 1, 3902, 1, 83775, 11123, 41373, 1, 7469, 3, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(shape, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]).unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape(shape, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }
}
