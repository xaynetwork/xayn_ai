pub mod encoding;
pub mod padding;
pub mod truncation;

use std::iter::once;

use displaydoc::Display;
use thiserror::Error;

use crate::{model::Vocab, normalizer::string::Offsets, post_tokenizer::encoding::Encoding};

/// A Bert post-tokenizer.
pub struct PostTokenizer {
    cls_id: u32,
    cls_token: String,
    sep_id: u32,
    sep_token: String,
}

/// The potential erros of the post-tokenizer.
#[derive(Debug, Display, Error)]
pub enum PostTokenizerError {
    /// Missing the class token in the vocabulary
    ClsToken,
    /// Missing the separation token in the vocabulary
    SepToken,
}

impl PostTokenizer {
    /// The number of added special tokens.
    pub(crate) const ADDED_TOKENS: usize = 2;

    /// Creates a Bert post-tokenizer.
    pub(crate) fn new(cls: String, sep: String, vocab: &Vocab) -> Result<Self, PostTokenizerError> {
        let cls_id = vocab
            .get(cls.as_str())
            .copied()
            .ok_or(PostTokenizerError::ClsToken)?;
        let sep_id = vocab
            .get(sep.as_str())
            .copied()
            .ok_or(PostTokenizerError::SepToken)?;

        Ok(PostTokenizer {
            cls_id,
            cls_token: cls,
            sep_id,
            sep_token: sep,
        })
    }

    /// Post-tokenizes the encoding.
    pub(crate) fn post_tokenize(&self, encoding: Encoding) -> Encoding {
        let len = encoding.len();
        let ids = once(self.cls_id)
            .chain(encoding.ids)
            .chain(once(self.sep_id))
            .collect();
        let type_ids = once(0).chain(encoding.type_ids).chain(once(0)).collect();
        let tokens = once(self.cls_token.clone())
            .chain(encoding.tokens)
            .chain(once(self.sep_token.clone()))
            .collect();
        let words = once(None).chain(encoding.words).chain(once(None)).collect();
        let offsets = once(Offsets(0, 0))
            .chain(encoding.offsets)
            .chain(once(Offsets(0, 0)))
            .collect();
        let special_tokens_mask = once(1)
            .chain(encoding.special_tokens_mask)
            .chain(once(1))
            .collect();
        let attention_mask = once(1)
            .chain(encoding.attention_mask)
            .chain(once(1))
            .collect();
        let overflowing = encoding.overflowing.map(|overflowing| {
            overflowing
                .into_iter()
                // the recursion is finite because overflowing is None for overflowed encodings
                .map(|encoding| self.post_tokenize(encoding))
                .collect()
        });
        // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't contain
        // the special tokens.
        let sequence_ranges = Some(once((0, 1..len + 1)).collect());

        Encoding {
            ids,
            type_ids,
            tokens,
            words,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
            sequence_ranges,
        }
    }
}
