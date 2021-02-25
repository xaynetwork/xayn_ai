use std::cmp::min;

use crate::encoding::Encoding;

/// A truncation strategy.
pub enum Truncation {
    /// No truncation.
    None,
    /// Fixed-length truncation.
    ///
    /// Configurable by:
    /// - `len`: Truncates to this length, must be greater or equal to the number of mandatory
    /// tokens added by the [`PostTokenizer`].
    /// - `stride`: Overlaps the truncated parts by this amount, must be zero or less than the
    /// length minus the number of mandatory tokens added by the [`PostTokenizer`].
    ///
    /// [`PostTokenizer`]: crate::PostTokenizer
    Fixed { len: usize, stride: usize },
}

impl Default for Truncation {
    fn default() -> Self {
        Self::None
    }
}

impl Truncation {
    pub fn truncate(&self, encoding: Encoding, added_tokens: usize) -> Encoding {
        match self {
            Self::None => encoding,
            Self::Fixed { len, stride } => {
                let len = (*len).checked_sub(added_tokens).unwrap_or_default();
                let stride = min(*stride, len.checked_sub(1).unwrap_or_default());
                encoding.truncate(len, stride)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizer::Offsets;

    fn truncation(len: usize) -> Truncation {
        Truncation::Fixed { len, stride: 0 }
    }

    fn encoding() -> Encoding {
        Encoding {
            ids: vec![1, 2, 3, 4],
            type_ids: vec![0, 0, 0, 0],
            tokens: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            words: vec![Some(0), Some(1), Some(2), Some(3)],
            offsets: vec![Offsets(0, 1), Offsets(1, 2), Offsets(2, 3), Offsets(3, 4)],
            special_tokens_mask: vec![0, 0, 0, 0],
            attention_mask: vec![1, 1, 1, 1],
            ..Encoding::default()
        }
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncation(3).truncate(encoding(), 0).len(), 3);
        assert_eq!(truncation(4).truncate(encoding(), 0).len(), 4);
        assert_eq!(truncation(5).truncate(encoding(), 0).len(), 4);
    }

    #[test]
    fn test_truncate_zero() {
        assert_eq!(truncation(0).truncate(encoding(), 0).len(), 0);
    }
}
