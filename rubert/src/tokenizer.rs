use std::{
    collections::HashMap,
    io::{BufRead, Error as IoError},
};

use derive_more::{Deref, From};
use displaydoc::Display;
use thiserror::Error;
use tokenizers::{
    models::wordpiece::WordPiece,
    normalizers::bert::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer,
    processors::bert::BertProcessing,
    tokenizer::{
        EncodeInput,
        Encoding,
        Error as BertTokenizerError,
        Model,
        Tokenizer as BertTokenizer,
    },
    utils::{
        padding::{PaddingDirection, PaddingParams, PaddingStrategy},
        truncation::{TruncationParams, TruncationStrategy},
    },
};

use crate::{
    ndarray::{Array1, ShapeError},
    utils::ArcArray2,
};

/// A [`RuBert`] tokenizer.
///
/// Wraps a pre-configured huggingface Bert tokenizer.
///
/// [`RuBert`]: crate::pipeline::RuBert
pub struct Tokenizer {
    tokenizer: BertTokenizer,
    token_size: usize,
}

/// Potential errors of the [`RuBert`] [`Tokenizer`].
///
/// [`RuBert`]: crate::pipeline::RuBert
#[derive(Debug, Display, Error)]
pub enum TokenizerError {
    /// Failed to read the vocabulary: {0}.
    Read(#[from] IoError),
    /// The vocabulary is missing the `[SEP]` token.
    SepToken,
    /// The vocabulary is missing the `[CLS]` token.
    ClsToken,
    /// Failed to encode sentences: {0}.
    Encoding(#[from] BertTokenizerError),
    /// Invalid encodings shapes: {0}.
    Shape(#[from] ShapeError),
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
    /// Can be set to strip accents and to lowercase the sentences. Requires the maximum number of
    /// tokens per tokenized sentence, which applies to padding and truncation and includes special
    /// tokens as well.
    ///
    /// # Errors
    /// Fails if the Bert model can't be build from the vocabulary file.
    ///
    /// [`RuBert`]: crate::pipeline::RuBert
    pub fn new(
        // `BufRead` instead of `AsRef<Path>` is needed for wasm
        vocab: impl BufRead,
        strip_accents: bool,
        lowercase: bool,
        token_size: usize,
    ) -> Result<Self, TokenizerError> {
        let vocab: HashMap<String, u32> = vocab
            .lines()
            .enumerate()
            .map(|(index, line)| line.map(|line| (line, index as u32)))
            .collect::<Result<_, _>>()?;

        let model = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .continuing_subword_prefix("##".into())
            .max_input_chars_per_word(100)
            // this is an `IoError` internally, we can't be more specific here due to how tokenizers
            // handles its errors internally, this will change once we have extracted the tokenizer
            .build()?;
        let sep_token = "[SEP]";
        let sep_id = model
            .token_to_id(sep_token)
            .ok_or(TokenizerError::SepToken)?;
        let cls_token = "[CLS]";
        let cls_id = model
            .token_to_id(cls_token)
            .ok_or(TokenizerError::ClsToken)?;

        let normalizer = BertNormalizer::new(true, true, Some(strip_accents), lowercase);
        let pre_tokenizer = BertPreTokenizer;
        let post_processor =
            BertProcessing::new((sep_token.into(), sep_id), (cls_token.into(), cls_id));

        let truncation = Some(TruncationParams {
            max_length: token_size,
            strategy: TruncationStrategy::LongestFirst,
            stride: 1,
        });
        let padding = Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(token_size),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
        });

        let mut tokenizer = BertTokenizer::new(model);
        tokenizer
            .with_normalizer(normalizer)
            .with_pre_tokenizer(pre_tokenizer)
            .with_post_processor(post_processor)
            .with_truncation(truncation)
            .with_padding(padding);

        Ok(Self {
            tokenizer,
            token_size,
        })
    }

    /// Encodes the sentences.
    ///
    /// # Errors
    /// Fails if any of the normalization, tokenization, pre- or post-processing steps fail.
    ///
    /// [`RuBert`]: crate::pipeline::RuBert
    pub fn encode<'s>(
        &self,
        sentences: Vec<impl Into<EncodeInput<'s>> + Send>,
    ) -> Result<Encodings, TokenizerError> {
        let batch_size = sentences.len();
        let encodings = self.tokenizer.encode_batch(sentences, true)?;
        let encodings = Encoding::merge(encodings, true);

        let input_ids = encodings
            .get_ids()
            .iter()
            .copied()
            .collect::<Array1<u32>>()
            .into_shape((batch_size, self.token_size))?
            .into_shared()
            .into();
        let attention_masks = encodings
            .get_attention_mask()
            .iter()
            .copied()
            .collect::<Array1<u32>>()
            .into_shape((batch_size, self.token_size))?
            .into_shared()
            .into();
        let token_type_ids = encodings
            .get_type_ids()
            .iter()
            .copied()
            .collect::<Array1<u32>>()
            .into_shape((batch_size, self.token_size))?
            .into_shared()
            .into();

        Ok(Encodings {
            input_ids,
            attention_masks,
            token_type_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use super::*;
    use crate::{ndarray::ArrayView, VOCAB};

    fn tokenizer(token_size: usize) -> Tokenizer {
        let vocab = BufReader::new(File::open(VOCAB).unwrap());
        let strip_accents = true;
        let lowercase = true;
        Tokenizer::new(vocab, strip_accents, lowercase, token_size).unwrap()
    }

    #[test]
    fn test_encode() {
        // too short
        let encoding = tokenizer(20)
            .encode(vec!["These are normal, common EMBEDDINGS.".to_string()])
            .unwrap();
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
        let encoding = tokenizer(10)
            .encode(vec!["These are normal, common EMBEDDINGS.".to_string()])
            .unwrap();
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
        let encoding = tokenizer(10)
            .encode(vec!["a b c".to_string(), "a b c d".to_string()])
            .unwrap();
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
        let encoding = tokenizer(10)
            .encode(vec![
                "a b c".to_string(),
                "a b c d e f g h i j k l".to_string(),
            ])
            .unwrap();
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
        let encoding = tokenizer(10)
            .encode(vec![
                "a b c d e f g h i j k l".to_string(),
                "a b c d e f g h i j k l m n".to_string(),
            ])
            .unwrap();
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
        let encoding = tokenizer(5)
            .encode(vec!["a b".to_string(), "a b c".to_string()])
            .unwrap();
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
        let encoding = tokenizer(5)
            .encode(vec!["a b c".to_string(), "a b c d".to_string()])
            .unwrap();
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
        let encoding = tokenizer(15)
            .encode(vec![
                "for “life-threatening storm surge” according".to_string()
            ])
            .unwrap();
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
