use std::iter::{once, repeat};

use crate::{encoding::Encoding, normalizer::Offsets};

pub struct PostTokenizer {
    cls_token: String,
    cls_id: u32,
    sep_token: String,
    sep_id: u32,
}

impl Default for PostTokenizer {
    fn default() -> Self {
        Self {
            cls_token: "[CLS]".into(),
            cls_id: 101,
            sep_token: "[SEP]".into(),
            sep_id: 102,
        }
    }
}

impl PostTokenizer {
    pub(crate) const ADDED_TOKENS: usize = 2;

    // TODO: get ids from vocab
    pub fn new(cls_token: String, cls_id: u32, sep_token: String, sep_id: u32) -> Self {
        Self {
            cls_token,
            cls_id,
            sep_token,
            sep_id,
        }
    }

    pub(crate) fn process(&self, encoding: Encoding, add_special_tokens: bool) -> Encoding {
        if !add_special_tokens {
            return encoding;
        }

        let len = encoding.ids.len();
        let len_added = len + Self::ADDED_TOKENS;
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
        let special_tokens_mask = once(1).chain(repeat(0).take(len)).chain(once(1)).collect();
        let attention_mask = vec![1; len_added];
        let overflowing = encoding.overflowing.map(|overflowing| {
            overflowing
                .into_iter()
                // the recursion is finite because overflowing is None for overflowed encodings
                .map(|encoding| self.process(encoding, add_special_tokens))
                .collect()
        });
        // For compatibility with `TemplateProcessing`, the sequence_ranges shouldn't contain
        // the special tokens.
        let sequence_ranges = Some(once((0, 1..len_added - 1)).collect());

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
