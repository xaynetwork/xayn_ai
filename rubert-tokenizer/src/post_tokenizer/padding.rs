use displaydoc::Display;
use num_traits::Num;
use thiserror::Error;

use crate::{model::Vocab, post_tokenizer::encoding::Encoding, SmallString};

/// A padding strategy.
pub struct Padding<N>(Paddings<N>);

/// The potential errors of the padding strategy.
#[derive(Debug, Display, Error)]
pub enum PaddingError {
    /// Missing the padding token in the vocabulary
    PadToken,
}

/// The available padding strategies.
enum Paddings<N> {
    /// No padding.
    None,
    /// Padding to a fixed length.
    Fixed {
        len: usize,
        pad_id: N,
        pad_token: SmallString,
    },
}

impl<N> Padding<N> {
    /// Creates an inert padding strategy.
    pub fn none() -> Self {
        Self(Paddings::None)
    }

    /// Creates a fixed-length padding strategy.
    pub fn fixed(len: usize, pad: impl AsRef<str>) -> Self
    where
        N: Num,
    {
        Self(Paddings::Fixed {
            len,
            pad_id: N::zero(),
            pad_token: pad.as_ref().into(),
        })
    }

    /// Validates this strategy.
    pub(crate) fn validate(mut self, vocab: &Vocab<N>) -> Result<Self, PaddingError>
    where
        N: Copy,
    {
        match self.0 {
            Paddings::None => Ok(self),
            Paddings::Fixed {
                ref mut pad_id,
                ref pad_token,
                ..
            } => {
                *pad_id = vocab
                    .get(pad_token.as_str())
                    .copied()
                    .ok_or(PaddingError::PadToken)?;
                Ok(self)
            }
        }
    }

    /// Pads the encoding.
    pub(crate) fn pad(&self, encoding: Encoding<N>) -> Encoding<N>
    where
        N: Num + Copy,
    {
        match self.0 {
            Paddings::None => encoding,
            Paddings::Fixed {
                len,
                pad_id,
                ref pad_token,
            } => encoding.pad(len, pad_id, N::zero(), pad_token),
        }
    }

    /// Gets the padding token.
    pub(crate) fn pad_token(&self) -> &str {
        match self.0 {
            Paddings::None => "",
            Paddings::Fixed { ref pad_token, .. } => pad_token.as_str(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoding(len: u32) -> Encoding<u32> {
        let mut encoding = Encoding::with_capacity(0);
        encoding.ids = (0..len).collect();
        encoding
    }

    fn fixed(len: usize) -> Padding<u32> {
        Padding::fixed(len, "[PAD]")
            .validate(&std::iter::once(("[PAD]".into(), 0)).collect::<Vocab<u32>>())
            .unwrap()
    }

    #[test]
    fn test_padding() {
        assert_eq!(fixed(3).pad(encoding(3)).len(), 3);
        assert_eq!(fixed(3).pad(encoding(5)).len(), 5);

        assert_eq!(fixed(5).pad(encoding(3)).len(), 5);
        assert_eq!(fixed(5).pad(encoding(5)).len(), 5);

        assert_eq!(fixed(7).pad(encoding(3)).len(), 7);
        assert_eq!(fixed(7).pad(encoding(5)).len(), 7);
    }
}
