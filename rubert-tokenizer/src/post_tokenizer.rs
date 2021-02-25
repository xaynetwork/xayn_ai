use std::iter::{once, repeat};

use crate::{encoding::Encoding, normalizer::Offsets};

/// A post-tokenizer.
pub enum PostTokenizer {
    /// No post-tokenization.
    None,
    /// Bert post-tokenization.
    Bert {
        // TODO: get ids from vocab
        cls_token: String,
        cls_id: u32,
        sep_token: String,
        sep_id: u32,
    },
}

impl Default for PostTokenizer {
    fn default() -> Self {
        Self::None
    }
}

impl PostTokenizer {
    // TODO: check and use `special_tokens_mask`, `attention_mask` and `sequence_ranges`
    pub(crate) fn process(&self, encoding: Encoding) -> Encoding {
        match self {
            Self::None => encoding,
            Self::Bert {
                cls_token,
                cls_id,
                sep_token,
                sep_id,
            } => {
                let len = encoding.len();
                let ids = once(*cls_id)
                    .chain(encoding.ids)
                    .chain(once(*sep_id))
                    .collect();
                let type_ids = once(0).chain(encoding.type_ids).chain(once(0)).collect();
                let tokens = once(cls_token.clone())
                    .chain(encoding.tokens)
                    .chain(once(sep_token.clone()))
                    .collect();
                let words = once(None).chain(encoding.words).chain(once(None)).collect();
                let offsets = once(Offsets(0, 0))
                    .chain(encoding.offsets)
                    .chain(once(Offsets(0, 0)))
                    .collect();
                let special_tokens_mask =
                    once(1).chain(repeat(0).take(len)).chain(once(1)).collect();
                let attention_mask = vec![1; len + 2];
                let overflowing = encoding.overflowing.map(|overflowing| {
                    overflowing
                        .into_iter()
                        // the recursion is finite because overflowing is None for overflowed encodings
                        .map(|encoding| self.process(encoding))
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
    }
}
