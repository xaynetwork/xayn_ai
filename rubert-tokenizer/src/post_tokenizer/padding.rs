use displaydoc::Display;
use thiserror::Error;

use crate::{model::Vocab, post_tokenizer::encoding::Encoding, SmallString};

/// A padding strategy.
pub struct Padding(Paddings);

/// The potential errors of the padding strategy.
#[derive(Debug, Display, Error)]
pub enum PaddingError {
    /// Missing the padding token in the vocabulary
    PadToken,
}

/// The available padding strategies.
enum Paddings {
    /// No padding.
    None,
    /// Padding to a fixed length.
    Fixed {
        len: usize,
        pad_id: u32,
        pad_token: SmallString,
    },
}

impl Padding {
    /// Creates an inert padding strategy.
    pub fn none() -> Self {
        Self(Paddings::None)
    }

    /// Creates a fixed-length padding strategy.
    pub fn fixed(len: usize, pad: impl AsRef<str>) -> Self {
        Self(Paddings::Fixed {
            len,
            pad_id: 0,
            pad_token: pad.as_ref().into(),
        })
    }

    /// Validates this strategy.
    pub(crate) fn validate(mut self, vocab: &Vocab) -> Result<Self, PaddingError> {
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
    pub(crate) fn pad(&self, encoding: Encoding) -> Encoding {
        match self.0 {
            Paddings::None => encoding,
            Paddings::Fixed {
                len,
                pad_id,
                ref pad_token,
            } => encoding.pad(len, pad_id, 0, pad_token),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoding(len: u32) -> Encoding {
        let mut encoding = Encoding::with_capacity(0);
        encoding.ids = (0..len).collect();
        encoding
    }

    fn fixed(len: usize) -> Padding {
        Padding::fixed(len, "[PAD]")
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
