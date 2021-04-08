use std::io::BufRead;

use derive_more::{Deref, From};
use displaydoc::Display;
use rubert_tokenizer::{Builder, BuilderError, Padding, Tokenizer as BertTokenizer, Truncation};
use thiserror::Error;

use ndarray::{Array1, Array2, Axis};

/// A pre-configured Bert tokenizer.
pub struct Tokenizer {
    tokenizer: BertTokenizer<i64>,
    pub(crate) token_size: usize,
}

/// The potential errors of the tokenizer.
#[derive(Debug, Display, Error)]
pub enum TokenizerError {
    /// Failed to build the tokenizer: {0}
    Builder(#[from] BuilderError),
}

/// The token ids of the encoded sequence.
#[derive(Clone, Deref, From)]
pub struct TokenIds(pub Array2<i64>);

/// The attention mask of the encoded sequence.
#[derive(Clone, Deref, From)]
pub struct AttentionMask(pub Array2<i64>);

/// The type ids of the encoded sequence.
#[derive(Clone, Deref, From)]
pub struct TypeIds(pub Array2<i64>);

/// The encoded sequence.
pub struct Encoding {
    pub token_ids: TokenIds,
    pub attention_mask: AttentionMask,
    pub type_ids: TypeIds,
}

impl Tokenizer {
    /// Creates a tokenizer from a vocabulary.
    ///
    /// Can be set to keep accents and to lowercase the sequences. Requires the maximum number of
    /// tokens per tokenized sequence, which applies to padding and truncation and includes special
    /// tokens as well.
    pub fn new(
        // `BufRead` instead of `AsRef<Path>` is needed for wasm
        vocab: impl BufRead,
        accents: bool,
        lowercase: bool,
        token_size: usize,
    ) -> Result<Self, TokenizerError> {
        let tokenizer = Builder::new(vocab)?
            .with_normalizer(true, false, accents, lowercase)
            .with_model("[UNK]", "##", 100)
            .with_post_tokenizer("[CLS]", "[SEP]")
            .with_truncation(Truncation::fixed(token_size, 0))
            .with_padding(Padding::fixed(token_size, "[PAD]"))
            .build()?;

        Ok(Tokenizer {
            tokenizer,
            token_size,
        })
    }

    /// Encodes the sequence.
    ///
    /// The encoding is in correct shape for the model.
    pub fn encode(&self, sequence: impl AsRef<str>) -> Encoding {
        let encoding = self.tokenizer.encode(sequence);
        let (token_ids, type_ids, _, _, _, _, attention_mask, _) = encoding.into();

        let token_ids = Array1::from(token_ids).insert_axis(Axis(0)).into();
        let attention_mask = Array1::from(attention_mask).insert_axis(Axis(0)).into();
        let type_ids = Array1::from(type_ids).insert_axis(Axis(0)).into();

        Encoding {
            token_ids,
            attention_mask,
            type_ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayView;
    use std::{fs::File, io::BufReader};

    use super::*;
    use crate::tests::VOCAB;

    fn tokenizer(token_size: usize) -> Tokenizer {
        let vocab = BufReader::new(File::open(VOCAB).unwrap());
        let accents = false;
        let lowercase = true;
        Tokenizer::new(vocab, accents, lowercase, token_size).unwrap()
    }

    #[test]
    fn test_encode_short() {
        let shape = (1, 20);
        let encoding = tokenizer(shape.1).encode("These are normal, common EMBEDDINGS.");
        assert_eq!(
            encoding.token_ids.0,
            ArrayView::from_shape(
                shape,
                &[2, 4538, 2128, 8561, 1, 6541, 69469, 2762, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_mask.0,
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
    }

    #[test]
    fn test_encode_long() {
        let shape = (1, 10);
        let encoding = tokenizer(shape.1).encode("These are normal, common EMBEDDINGS.");
        assert_eq!(
            encoding.token_ids.0,
            ArrayView::from_shape(shape, &[2, 4538, 2128, 8561, 1, 6541, 69469, 2762, 5, 3])
                .unwrap(),
        );
        assert_eq!(
            encoding.attention_mask.0,
            ArrayView::from_shape(shape, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap(),
        );
        assert_eq!(
            encoding.type_ids.0,
            ArrayView::from_shape(shape, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }

    #[test]
    fn test_encode_troublemakers() {
        let shape = (1, 15);
        let encoding = tokenizer(shape.1).encode("for “life-threatening storm surge” according");
        assert_eq!(
            encoding.token_ids.0,
            ArrayView::from_shape(
                shape,
                &[2, 1665, 1, 3902, 1, 83775, 11123, 41373, 1, 7469, 3, 0, 0, 0, 0],
            )
            .unwrap(),
        );
        assert_eq!(
            encoding.attention_mask.0,
            ArrayView::from_shape(shape, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]).unwrap(),
        );
        assert_eq!(
            encoding.type_ids.0,
            ArrayView::from_shape(shape, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap(),
        );
    }
}
