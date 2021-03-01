use std::iter::{once, repeat};

use anyhow::bail;

use crate::{
    model::{encoding::Encoding, Vocab},
    normalizer::string::Offsets,
    Error,
};

/// A post-tokenizer.
///
/// Defaults to the [`none()`] post-tokenizer.
pub struct PostTokenizer(PostTokenizers);

/// The post-tokenizers.
enum PostTokenizers {
    /// No post-tokenization.
    None,
    /// Bert post-tokenization.
    Bert {
        cls_id: u32,
        cls_token: String,
        sep_id: u32,
        sep_token: String,
    },
}

impl Default for PostTokenizer {
    fn default() -> Self {
        Self::none()
    }
}

impl PostTokenizer {
    /// Creates an inert post-tokenizer.
    pub fn none() -> Self {
        Self(PostTokenizers::None)
    }

    /// Creates a Bert post-tokenizer.
    ///
    /// The tokens must be part of the vocabulary.
    pub fn bert(cls: impl Into<String>, sep: impl Into<String>) -> Self {
        Self(PostTokenizers::Bert {
            cls_id: 0,
            cls_token: cls.into(),
            sep_id: 0,
            sep_token: sep.into(),
        })
    }

    /// Validates itself.
    pub(crate) fn validate(mut self, vocab: &Vocab) -> Result<Self, Error> {
        match self.0 {
            PostTokenizers::None => Ok(self),
            PostTokenizers::Bert {
                ref mut cls_id,
                ref cls_token,
                ref mut sep_id,
                ref sep_token,
            } => {
                if let Some(id) = vocab.get(cls_token) {
                    *cls_id = *id;
                } else {
                    bail!("class token doesn't exist in the vocab");
                };
                if let Some(id) = vocab.get(sep_token) {
                    *sep_id = *id;
                } else {
                    bail!("separation token doesn't exist in the vocab");
                };
                Ok(self)
            }
        }
    }

    pub(crate) const fn added_tokens(&self) -> usize {
        match self.0 {
            PostTokenizers::None => 0,
            PostTokenizers::Bert { .. } => 2,
        }
    }

    pub(crate) fn process(&self, encoding: Encoding) -> Encoding {
        match self.0 {
            PostTokenizers::None => encoding,
            PostTokenizers::Bert {
                cls_id,
                ref cls_token,
                sep_id,
                ref sep_token,
            } => {
                let len = encoding.len();
                let ids = once(cls_id)
                    .chain(encoding.ids)
                    .chain(once(sep_id))
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
