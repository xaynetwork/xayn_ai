use std::{collections::HashMap, io::BufRead};

use tokenizers::{
    models::wordpiece::WordPiece,
    normalizers::bert::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer,
    processors::bert::BertProcessing,
    tokenizer::{EncodeInput, Encoding, Model, Result as TokenizerResult, Tokenizer},
    utils::{
        padding::{PaddingDirection, PaddingParams, PaddingStrategy},
        truncation::{TruncationParams, TruncationStrategy},
    },
};

use crate::{ndarray::Array1, utils::ArcArray2};

#[derive(Debug)]
/// Tokenization errors.
pub enum RuBertTokenizerError {
    MissingSepToken,
    MissingClsToken,
}

impl std::error::Error for RuBertTokenizerError {}

impl std::fmt::Display for RuBertTokenizerError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        let tok_str = match self {
            RuBertTokenizerError::MissingSepToken => "[SEP]",
            RuBertTokenizerError::MissingClsToken => "[CLS]",
        };

        write!(
            fmt,
            "Tokenizer error: Missing {} token from the vocabulary",
            tok_str
        )
    }
}

// A pre-configured wrapper around the huggingface tokenizer.
pub struct RuBertTokenizer {
    tokenizer: Tokenizer,
    tokens_size: usize,
}

impl RuBertTokenizer {
    /// Creates a new Bert tokenizer from a `vocab`ulary file.
    ///
    /// Can be set to `strip_accents` and `lowercase` the sentences. Requires the maximum number of
    /// tokens per tokenized sentence (`tokens_size`), which applies to padding and truncation and
    /// includes special tokens as well.
    ///
    /// # Errors
    /// Fails if the wordpiece model can't be build from the vocabulary file.
    pub fn new(
        vocab: impl BufRead,
        strip_accents: bool,
        lowercase: bool,
        tokens_size: usize,
    ) -> TokenizerResult<Self> {
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
            .build()?;
        let sep = "[SEP]";
        let sep_id = model
            .token_to_id(sep)
            .ok_or(RuBertTokenizerError::MissingSepToken)?;
        let cls = "[CLS]";
        let cls_id = model
            .token_to_id(cls)
            .ok_or(RuBertTokenizerError::MissingClsToken)?;

        let normalizer = BertNormalizer::new(true, true, Some(strip_accents), lowercase);
        let pre_tokenizer = BertPreTokenizer;
        let post_processor = BertProcessing::new((sep.into(), sep_id), (cls.into(), cls_id));

        let truncation = Some(TruncationParams {
            max_length: tokens_size,
            strategy: TruncationStrategy::LongestFirst,
            stride: 1,
        });
        let padding = Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(tokens_size),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
        });

        let mut tokenizer = Tokenizer::new(model);
        tokenizer
            .with_normalizer(normalizer)
            .with_pre_tokenizer(pre_tokenizer)
            .with_post_processor(post_processor)
            .with_truncation(truncation)
            .with_padding(padding);

        Ok(Self {
            tokenizer,
            tokens_size,
        })
    }

    /// Encodes the `sentences` with the Bert tokenizer.
    ///
    /// The encodings are a triple of `input_ids`, `attention_masks` and `token_type_ids`.
    ///
    /// # Errors
    /// Fails if any of the normalization, tokenization, pre- or post-processing steps fails.
    pub fn encode<'s>(
        &self,
        sentences: Vec<impl Into<EncodeInput<'s>> + Send>,
    ) -> TokenizerResult<(ArcArray2<u32>, ArcArray2<u32>, ArcArray2<u32>)> {
        let batch_size = sentences.len();
        let encodings = self.tokenizer.encode_batch(sentences, true)?;
        let encodings = Encoding::merge(encodings, true);

        let input_ids = encodings
            .get_ids()
            .iter()
            .copied()
            .collect::<Array1<u32>>()
            .into_shape((batch_size, self.tokens_size))
            .unwrap()
            .into_shared();
        let attention_masks = encodings
            .get_attention_mask()
            .iter()
            .copied()
            .collect::<Array1<u32>>()
            .into_shape((batch_size, self.tokens_size))
            .unwrap()
            .into_shared();
        let token_type_ids = encodings
            .get_type_ids()
            .iter()
            .copied()
            .collect::<Array1<u32>>()
            .into_shape((batch_size, self.tokens_size))
            .unwrap()
            .into_shared();

        Ok((input_ids, attention_masks, token_type_ids))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use crate::ndarray::ArrayView;

    use super::*;

    fn tokenizer(token_size: usize) -> RuBertTokenizer {
        let vocab = "../assets/rubert/rubert-uncased.txt";
        let vocab = BufReader::new(File::open(vocab).unwrap());
        RuBertTokenizer::new(vocab, true, true, token_size).unwrap()
    }

    #[test]
    fn test_encode() {
        // too short
        let (input_ids, attention_masks, token_type_ids) = tokenizer(20)
            .encode(vec!["These are normal, common EMBEDDINGS.".to_string()])
            .unwrap();
        assert_eq!(
            input_ids,
            ArrayView::from_shape(
                (1, 20),
                &[2, 4538, 2128, 8561, 1, 6541, 69469, 2762, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            attention_masks,
            ArrayView::from_shape(
                (1, 20),
                &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            token_type_ids,
            ArrayView::from_shape(
                (1, 20),
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );

        // too long
        let (input_ids, attention_masks, token_type_ids) = tokenizer(10)
            .encode(vec!["These are normal, common EMBEDDINGS.".to_string()])
            .unwrap();
        assert_eq!(
            input_ids,
            ArrayView::from_shape((1, 10), &[2, 4538, 2128, 8561, 1, 6541, 69469, 2762, 5, 3])
                .unwrap(),
        );
        assert_eq!(
            attention_masks,
            ArrayView::from_shape((1, 10), &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap(),
        );
        assert_eq!(
            token_type_ids,
            ArrayView::from_shape((1, 10), &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }

    #[test]
    fn test_encode_batch() {
        // both too short
        let (input_ids, attention_masks, token_type_ids) = tokenizer(10)
            .encode(vec!["a b c".to_string(), "a b c d".to_string()])
            .unwrap();
        assert_eq!(
            input_ids,
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
            attention_masks,
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
            token_type_ids,
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
        let (input_ids, attention_masks, token_type_ids) = tokenizer(10)
            .encode(vec![
                "a b c".to_string(),
                "a b c d e f g h i j k l".to_string(),
            ])
            .unwrap();
        assert_eq!(
            input_ids,
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
            attention_masks,
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
            token_type_ids,
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
        let (input_ids, attention_masks, token_type_ids) = tokenizer(10)
            .encode(vec![
                "a b c d e f g h i j k l".to_string(),
                "a b c d e f g h i j k l m n".to_string(),
            ])
            .unwrap();
        assert_eq!(
            input_ids,
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
            attention_masks,
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
            token_type_ids,
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
        let (input_ids, attention_masks, token_type_ids) = tokenizer(5)
            .encode(vec!["a b".to_string(), "a b c".to_string()])
            .unwrap();
        assert_eq!(
            input_ids,
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
            attention_masks,
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
            token_type_ids,
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
        let (input_ids, attention_masks, token_type_ids) = tokenizer(5)
            .encode(vec!["a b c".to_string(), "a b c d".to_string()])
            .unwrap();
        assert_eq!(
            input_ids,
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
            attention_masks,
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
            token_type_ids,
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
        let (input_ids, attention_masks, token_type_ids) = tokenizer(15)
            .encode(vec![
                "for “life-threatening storm surge” according".to_string()
            ])
            .unwrap();
        assert_eq!(
            input_ids,
            ArrayView::from_shape(
                (1, 15),
                &[2, 1665, 1, 3902, 1, 83775, 11123, 41373, 1, 7469, 3, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            attention_masks,
            ArrayView::from_shape((1, 15), &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]).unwrap(),
        );
        assert_eq!(
            token_type_ids,
            ArrayView::from_shape((1, 15), &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }
}
