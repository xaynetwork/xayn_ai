use std::io::BufRead;

use derive_more::{Deref, From};
use displaydoc::Display;
use rubert_tokenizer::{Builder, BuilderError, Padding, Tokenizer as BertTokenizer, Truncation};
use thiserror::Error;

use crate::utils::ArcArray2;

/// A [`RuBert`] tokenizer.
///
/// Wraps a pre-configured Bert tokenizer.
///
/// [`RuBert`]: crate::pipeline::RuBert
pub struct Tokenizer(BertTokenizer<u32>);

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
pub struct InputIds(pub(crate) ArcArray2<u32>);

/// The attention masks of the encoded sentences.
#[derive(Clone, Deref, From)]
pub struct AttentionMasks(pub(crate) ArcArray2<u32>);

/// The token type ids of the encoded sentences.
#[derive(Clone, Deref, From)]
pub struct TokenTypeIds(pub(crate) ArcArray2<u32>);

/// The encoded sentences.
pub struct Encodings {
    pub(crate) input_ids: InputIds,
    pub(crate) attention_masks: AttentionMasks,
    pub(crate) token_type_ids: TokenTypeIds,
}

impl Tokenizer {
    /// Creates a [`RuBert`] tokenizer from a vocabulary file.
    ///
    /// Can be set to strip accents and to lowercase the sequences. Requires the maximum number of
    /// tokens per tokenized sequence, which applies to padding and truncation and includes special
    /// tokens as well.
    ///
    /// # Errors
    /// Fails if the Bert model can't be build from the vocabulary file.
    ///
    /// [`RuBert`]: crate::pipeline::RuBert
    pub fn new(
        // `BufRead` instead of `AsRef<Path>` is needed for wasm
        vocab: impl BufRead,
        accents: bool,
        lowercase: bool,
        token_size: usize,
    ) -> Result<Self, TokenizerError> {
        Ok(Tokenizer(
            Builder::new(vocab)?
                .with_normalizer(true, false, accents, lowercase)
                .with_model("[UNK]", "##", 100)
                .with_post_tokenizer("[CLS]", "[SEP]")
                .with_truncation(Truncation::fixed(token_size, 0))
                .with_padding(Padding::fixed(token_size, "[PAD]"))
                .build()?,
        ))
    }

    /// Encodes the sequence.
    pub fn encode(&self, sequence: impl AsRef<str>) -> Encodings {
        let encoding = self.0.encode(sequence);
        let token_size = encoding.len();

        let input_ids = InputIds(ArcArray2::from_shape_fn((1, token_size), |(_, j)| {
            encoding.ids()[j]
        }));
        let attention_masks =
            AttentionMasks(ArcArray2::from_shape_fn((1, token_size), |(_, j)| {
                encoding.attention_mask()[j]
            }));
        let token_type_ids = TokenTypeIds(ArcArray2::from_shape_fn((1, token_size), |(_, j)| {
            encoding.type_ids()[j]
        }));

        Encodings {
            input_ids,
            attention_masks,
            token_type_ids,
        }
    }

    /// Encodes the sequences.
    pub fn encode_batch(&self, sequences: &[impl AsRef<str>]) -> Encodings {
        if sequences.is_empty() {
            let input_ids = InputIds(ArcArray2::default((0, 0)));
            let attention_masks = AttentionMasks(ArcArray2::default((0, 0)));
            let token_type_ids = TokenTypeIds(ArcArray2::default((0, 0)));

            Encodings {
                input_ids,
                attention_masks,
                token_type_ids,
            }
        } else {
            let encodings = self.0.encode_batch(sequences);
            let batch_size = encodings.len();
            let token_size = encodings[0].len();

            let input_ids = InputIds(ArcArray2::from_shape_fn(
                (batch_size, token_size),
                |(i, j)| encodings[i].ids()[j],
            ));
            let attention_masks = AttentionMasks(ArcArray2::from_shape_fn(
                (batch_size, token_size),
                |(i, j)| encodings[i].attention_mask()[j],
            ));
            let token_type_ids = TokenTypeIds(ArcArray2::from_shape_fn(
                (batch_size, token_size),
                |(i, j)| encodings[i].type_ids()[j],
            ));

            Encodings {
                input_ids,
                attention_masks,
                token_type_ids,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use super::*;
    use crate::{ndarray::ArrayView, VOCAB};

    fn tokenizer(token_size: usize) -> Tokenizer {
        let vocab = BufReader::new(File::open(VOCAB).unwrap());
        let accents = true;
        let lowercase = true;
        Tokenizer::new(vocab, accents, lowercase, token_size).unwrap()
    }

    #[test]
    fn test_encode() {
        // too short
        let encoding = tokenizer(20).encode("These are normal, common EMBEDDINGS.");
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                (1, 20),
                &[
                    2u32, 4538, 2128, 8561, 1, 6541, 69469, 2762, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0
                ],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape(
                (1, 20),
                &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape(
                (1, 20),
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );

        // too long
        let encoding = tokenizer(10).encode("These are normal, common EMBEDDINGS.");
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape((1, 10), &[2, 4538, 2128, 8561, 1, 6541, 69469, 2762, 5, 3])
                .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape((1, 10), &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape((1, 10), &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }

    #[test]
    fn test_encode_batch() {
        // both too short
        let encoding = tokenizer(10).encode_batch(&["a b c", "a b c d"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                (2, 10),
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
                (2, 10),
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
                (2, 10),
                &[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
            )
            .unwrap(),
        );

        // one too short and one too long
        let encoding = tokenizer(10).encode_batch(&["a b c", "a b c d e f g h i j k l"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                (2, 10),
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
                (2, 10),
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
                (2, 10),
                &[
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]
            )
            .unwrap(),
        );

        // both too long
        let encoding =
            tokenizer(10).encode_batch(&["a b c d e f g h i j k l", "a b c d e f g h i j k l m n"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                (2, 10),
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
                (2, 10),
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
                (2, 10),
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
        let encoding = tokenizer(5).encode_batch(&["a b", "a b c"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                (2, 5),
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
                (2, 5),
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
                (2, 5),
                &[
                    0, 0, 0, 0, 0, //
                    0, 0, 0, 0, 0,
                ]
            )
            .unwrap(),
        );

        // first fits, second 1 too long
        let encoding = tokenizer(5).encode_batch(&["a b c", "a b c d"]);
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                (2, 5),
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
                (2, 5),
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
                (2, 5),
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
        let encoding = tokenizer(15).encode("for “life-threatening storm surge” according");
        assert_eq!(
            encoding.input_ids.0,
            ArrayView::from_shape(
                (1, 15),
                &[2, 1665, 1, 3902, 1, 83775, 11123, 41373, 1, 7469, 3, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_masks.0,
            ArrayView::from_shape((1, 15), &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]).unwrap(),
        );
        assert_eq!(
            encoding.token_type_ids.0,
            ArrayView::from_shape((1, 15), &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }
}
