use displaydoc::Display;
use thiserror::Error;

use crate::post_tokenizer::{encoding::Encoding, ADDED_TOKENS};

/// A truncation strategy.
pub struct Truncation(Truncations);

/// The potential errors of the truncation strategy.
#[derive(Debug, Display, Error)]
pub enum TruncationError {
    /// Invalid truncation length, must be greater or equal to the number of special tokens added by
    /// the post-tokenizer
    FixedLength,
    /// Invalid truncation stride, must be zero or less than the truncation length minus the number
    /// of special tokens added by the post-tokenizer
    FixedStride,
}

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
    pub(crate) fn validate(self) -> Result<Self, TruncationError> {
        match self.0 {
            Truncations::None => Ok(self),
            Truncations::Fixed { len, stride } => {
                if len < ADDED_TOKENS {
                    Err(TruncationError::FixedLength)
                } else if stride != 0 && stride >= len - ADDED_TOKENS {
                    Err(TruncationError::FixedStride)
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
    pub(crate) fn truncate<N>(&self, encoding: Encoding<N>) -> Encoding<N>
    where
        N: Copy,
    {
        match self.0 {
            Truncations::None => encoding,
            Truncations::Fixed { len, stride } => encoding.truncate(len - ADDED_TOKENS, stride),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizer::string::Offsets;

    fn encoding() -> Encoding<u32> {
        Encoding {
            ids: vec![1, 2, 3, 4],
            type_ids: vec![0, 0, 0, 0],
            tokens: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            word_indices: vec![Some(0), Some(1), Some(2), Some(3)],
            offsets: vec![Offsets(0, 1), Offsets(1, 2), Offsets(2, 3), Offsets(3, 4)],
            special_tokens_mask: vec![0, 0, 0, 0],
            attention_mask: vec![1, 1, 1, 1],
            sequence_ranges: None,
            overflowing: None,
        }
    }

    fn fixed(len: usize) -> Truncation {
        Truncation::fixed(len, 0).validate().unwrap()
    }

    #[test]
    fn test_truncate() {
        assert_eq!(fixed(5).truncate(encoding()).len(), 3);
        assert_eq!(fixed(6).truncate(encoding()).len(), 4);
        assert_eq!(fixed(7).truncate(encoding()).len(), 4);
    }

    #[test]
    fn test_truncate_zero() {
        assert_eq!(fixed(2).truncate(encoding()).len(), 0);
    }
}
