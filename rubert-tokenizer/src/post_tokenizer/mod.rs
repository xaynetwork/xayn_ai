pub mod padding;
pub mod truncation;

use std::iter::once;

use anyhow::bail;

use crate::{
    model::{encoding::Encoding, Vocab},
    normalizer::string::Offsets,
    Error,
};

/// A Bert post-tokenizer.
pub struct PostTokenizer {
    cls_id: u32,
    cls_token: String,
    sep_id: u32,
    sep_token: String,
}

impl PostTokenizer {
    pub(crate) const ADDED_TOKENS: usize = 2;

    /// Validates itself.
    pub(crate) fn new(cls: String, sep: String, vocab: &Vocab) -> Result<Self, Error> {
        let cls_id = if let Some(id) = vocab.get(cls.as_str()) {
            *id
        } else {
            bail!("class token doesn't exist in the vocab");
        };
        let sep_id = if let Some(id) = vocab.get(sep.as_str()) {
            *id
        } else {
            bail!("separation token doesn't exist in the vocab");
        };

        Ok(PostTokenizer {
            cls_id,
            cls_token: cls,
            sep_id,
            sep_token: sep,
        })
    }

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
