use anyhow::anyhow;

use crate::{
    post_tokenizer::{encoding::Encoding, PostTokenizer},
    Error,
};

/// A truncation strategy.
pub struct Truncation(Truncations);

/// The available truncation strategies.
enum Truncations {
    /// No truncation.
    None,
    /// Truncation to a fixed length.
    Fixed { len: usize, stride: usize },
}

impl Truncation {
    /// Creates an inert truncation strategy.
    pub fn none() -> Self {
        Self(Truncations::None)
    }

    /// Creates a fixed-length truncation strategy.
    pub fn fixed(len: usize, stride: usize) -> Self {
        Self(Truncations::Fixed { len, stride })
    }

    /// Validates this strategy.
    pub(crate) fn validate(self) -> Result<Self, Error> {
        match self.0 {
            Truncations::None => Ok(self),
            Truncations::Fixed { len, stride } => {
                if len < PostTokenizer::ADDED_TOKENS {
                    Err(anyhow!("length must be greater or equal to the number of mandatory tokens added by the post-tokenizer"))
                } else if stride >= len - PostTokenizer::ADDED_TOKENS {
                    Err(anyhow!("stride must be zero or less than the length minus the number of mandatory tokens added by the post-tokenizer"))
                } else {
                    Ok(self)
                }
            }
        }
    }

    /// Truncates the encoding.
    ///
    /// # Panics
    /// May panic/underflow if the truncation strategy has not been validated.
    pub(crate) fn truncate(&self, encoding: Encoding, added_tokens: usize) -> Encoding {
        match self.0 {
            Truncations::None => encoding,
            Truncations::Fixed { len, stride } => encoding.truncate(len - added_tokens, stride),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizer::string::Offsets;

    fn encoding() -> Encoding {
        Encoding {
            ids: vec![1, 2, 3, 4],
            type_ids: vec![0, 0, 0, 0],
            tokens: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            words: vec![Some(0), Some(1), Some(2), Some(3)],
            offsets: vec![Offsets(0, 1), Offsets(1, 2), Offsets(2, 3), Offsets(3, 4)],
            special_tokens_mask: vec![0, 0, 0, 0],
            attention_mask: vec![1, 1, 1, 1],
            sequence_ranges: None,
            overflowing: None,
        }
    }

    fn fixed(len: usize) -> Truncation {
        Truncation::fixed(len, 0)
    }

    #[test]
    fn test_truncate() {
        assert_eq!(fixed(3).truncate(encoding(), 0).len(), 3);
        assert_eq!(fixed(4).truncate(encoding(), 0).len(), 4);
        assert_eq!(fixed(5).truncate(encoding(), 0).len(), 4);
    }

    #[test]
    fn test_truncate_zero() {
        assert_eq!(fixed(0).truncate(encoding(), 0).len(), 0);
    }
}
