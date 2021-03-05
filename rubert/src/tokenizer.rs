use std::io::BufRead;

use derive_more::{Deref, From};
use displaydoc::Display;
use rubert_tokenizer::{Builder, BuilderError, Padding, Tokenizer as BertTokenizer, Truncation};
use thiserror::Error;

use crate::ndarray::{Array2, Dim, Ix2};

/// A wrapped, pre-configured Bert tokenizer.
pub struct Tokenizer {
    tokenizer: BertTokenizer<i64>,
    shape: Ix2,
}

/// The potential errors of the tokenizer.
#[derive(Debug, Display, Error)]
pub enum TokenizerError {
    /// Failed to build the tokenizer: {0}
    Builder(#[from] BuilderError),
}

/// The token ids of the encoded sequences.
#[derive(Clone, Deref, From)]
pub struct TokenIds(pub Array2<i64>);

/// The attention masks of the encoded sequences.
#[derive(Clone, Deref, From)]
pub struct AttentionMasks(pub Array2<i64>);

/// The type ids of the encoded sequences.
#[derive(Clone, Deref, From)]
pub struct TypeIds(pub Array2<i64>);

/// The encoded sequences.
pub struct Encodings {
    pub token_ids: TokenIds,
    pub attention_masks: AttentionMasks,
    pub type_ids: TypeIds,
}

impl Tokenizer {
    /// Creates a tokenizer from a vocabulary.
    ///
    /// Can be set to strip accents and to lowercase the sequences. Requires the maximum number of
    /// sequences as well as tokens per tokenized sequence, which applies to padding and truncation
    /// and includes special tokens as well.
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
        let shape = Dim([batch_size, token_size]);

        Ok(Tokenizer { tokenizer, shape })
    }

    /// Encodes the sequence.
    ///
    /// The encoding is in correct shape for the model.
    pub fn encode(&self, sequence: impl AsRef<str>) -> Encodings {
        let encoding = self.tokenizer.encode(sequence);

        let token_ids = Array2::from_shape_fn(self.shape, |(i, j)| {
            encoding.ids().get(i + j).copied().unwrap_or(0)
        })
        .into();
        let attention_masks = Array2::from_shape_fn(self.shape, |(i, j)| {
            encoding.attention_mask().get(i + j).copied().unwrap_or(0)
        })
        .into();
        let type_ids = Array2::from_shape_fn(self.shape, |(i, j)| {
            encoding.type_ids().get(i + j).copied().unwrap_or(0)
        })
        .into();

        Encodings {
            token_ids,
            attention_masks,
            type_ids,
        }
    }

    /// Encodes the batch of sequences.
    ///
    /// The encodings are in correct shape for the model.
    pub fn encode_batch(&self, sequences: &[impl AsRef<str>]) -> Encodings {
        let encodings = self.tokenizer.encode_batch(sequences);

        let token_ids = Array2::from_shape_fn(self.shape, |(i, j)| {
            encodings
                .get(i)
                .map(|encoding| encoding.ids().get(j))
                .flatten()
                .copied()
                .unwrap_or(0)
        })
        .into();
        let attention_masks = Array2::from_shape_fn(self.shape, |(i, j)| {
            encodings
                .get(i)
                .map(|encoding| encoding.attention_mask().get(j))
                .flatten()
                .copied()
                .unwrap_or(0)
        })
        .into();
        let type_ids = Array2::from_shape_fn(self.shape, |(i, j)| {
            encodings
                .get(i)
                .map(|encoding| encoding.type_ids().get(j))
                .flatten()
                .copied()
                .unwrap_or(0)
        })
        .into();

        Encodings {
            token_ids,
            attention_masks,
            type_ids,
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
        let encoding = tokenizer(shape).encode("These are normal, common EMBEDDINGS.");
        assert_eq!(
            encoding.token_ids.0,
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
            encoding.type_ids.0,
            ArrayView::from_shape(
                shape,
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );

        // too long
        let shape = (1, 10);
        let encoding = tokenizer(shape).encode("These are normal, common EMBEDDINGS.");
        assert_eq!(
            encoding.token_ids.0,
            ArrayView::from_shape(shape, &[2, 4538, 2128, 8561, 1, 6541, 69469, 2762, 5, 3])
                .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(shape, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap(),
        );
        assert_eq!(
            encoding.type_ids.0,
            ArrayView::from_shape(shape, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }

    #[test]
    fn test_encode_batch() {
        // both too short
        let shape = (2, 10);
        let encoding = tokenizer(shape).encode_batch(&["a b c", "a b c d"]);
        assert_eq!(
            encoding.token_ids.0,
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
            encoding.type_ids.0,
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
        let encoding = tokenizer(shape).encode_batch(&["a b c", "a b c d e f g h i j k l"]);
        assert_eq!(
            encoding.token_ids.0,
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
            encoding.type_ids.0,
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
        let encoding = tokenizer(shape)
            .encode_batch(&["a b c d e f g h i j k l", "a b c d e f g h i j k l m n"]);
        assert_eq!(
            encoding.token_ids.0,
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
            encoding.type_ids.0,
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
        let encoding = tokenizer(shape).encode_batch(&["a b", "a b c"]);
        assert_eq!(
            encoding.token_ids.0,
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
            encoding.type_ids.0,
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
        let encoding = tokenizer(shape).encode_batch(&["a b c", "a b c d"]);
        assert_eq!(
            encoding.token_ids.0,
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
            encoding.type_ids.0,
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
        let encoding = tokenizer(shape).encode("for “life-threatening storm surge” according");
        assert_eq!(
            encoding.token_ids.0,
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
            encoding.type_ids.0,
            ArrayView::from_shape(shape, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }
}
