pub mod encoding;
pub mod padding;
pub mod truncation;

use std::iter::once;

use displaydoc::Display;
use num_traits::Num;
use thiserror::Error;

use crate::{
    model::Vocab,
    normalizer::string::Offsets,
    post_tokenizer::encoding::Encoding,
    SmallString,
};

/// A Bert post-tokenizer.
#[derive(Debug)]
pub struct PostTokenizer<N> {
    cls_id: N,
    pub(crate) cls_token: SmallString,
    sep_id: N,
    pub(crate) sep_token: SmallString,
}

/// The potential erros of the post-tokenizer.
#[derive(Debug, Display, Error, PartialEq)]
pub enum PostTokenizerError {
    /// Missing the class token in the vocabulary
    ClsToken,
    /// Missing the separation token in the vocabulary
    SepToken,
}

/// The number of added special tokens.
pub const ADDED_TOKENS: usize = 2;

impl<N> PostTokenizer<N>
where
    N: Copy,
{
    /// Creates a Bert post-tokenizer.
    pub(crate) fn new(
        cls_token: SmallString,
        sep_token: SmallString,
        vocab: &Vocab<N>,
    ) -> Result<Self, PostTokenizerError> {
        let cls_id = vocab
            .get(cls_token.as_str())
            .copied()
            .ok_or(PostTokenizerError::ClsToken)?;
        let sep_id = vocab
            .get(sep_token.as_str())
            .copied()
            .ok_or(PostTokenizerError::SepToken)?;

        Ok(PostTokenizer {
            cls_id,
            cls_token,
            sep_id,
            sep_token,
        })
    }

    /// Post-tokenizes the encoding.
    pub(crate) fn post_tokenize(&self, encoding: Encoding<N>) -> Encoding<N>
    where
        N: Num,
    {
        let len = encoding.len();
        let ids = once(self.cls_id)
            .chain(encoding.ids)
            .chain(once(self.sep_id))
            .collect();
        let type_ids = once(N::zero())
            .chain(encoding.type_ids)
            .chain(once(N::zero()))
            .collect();
        let tokens = once(self.cls_token.to_string())
            .chain(encoding.tokens)
            .chain(once(self.sep_token.to_string()))
            .collect();
        let word_indices = once(None)
            .chain(encoding.word_indices)
            .chain(once(None))
            .collect();
        let cls_token_offset = encoding
            .offsets
            .first()
            .map(|&Offsets(start, _)| Offsets(start, start))
            .unwrap_or_default();
        let sep_token_offset = encoding
            .offsets
            .last()
            .map(|&Offsets(_, end)| Offsets(end, end))
            .unwrap_or_default();
        let offsets = once(cls_token_offset)
            .chain(encoding.offsets)
            .chain(once(sep_token_offset))
            .collect();
        let special_tokens_mask = once(N::one())
            .chain(encoding.special_tokens_mask)
            .chain(once(N::one()))
            .collect();
        let attention_mask = once(N::one())
            .chain(encoding.attention_mask)
            .chain(once(N::one()))
            .collect();
        let overflowing = encoding.overflowing.map(|overflowing| {
            overflowing
                .into_iter()
                // the recursion is finite because overflowing is None for overflowed encodings
                .map(|encoding| self.post_tokenize(encoding))
                .collect()
        });
        let sequence_ranges = Some(once((0, 1..len + 1)).collect());

        Encoding {
            ids,
            type_ids,
            tokens,
            word_indices,
            offsets,
            special_tokens_mask,
            attention_mask,
            sequence_ranges,
            overflowing,
        }
    }
}
