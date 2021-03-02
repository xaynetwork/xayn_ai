use anyhow::anyhow;

use crate::{
    model::{encoding::Encoding, Vocab},
    Error,
};

/// A padding strategy.
///
/// Defaults to the [`none()`] padding strategy.
pub struct Padding(Paddings);

/// The padding strategies.
enum Paddings {
    /// No padding.
    None,
    /// Padding to a fixed length.
    Fixed {
        len: usize,
        pad_id: u32,
        pad_token: String,
    },
}

impl Padding {
    /// Creates an inert padding strategy.
    pub fn none() -> Self {
        Self(Paddings::None)
    }

    /// Creates a fixed-length padding strategy.
    ///
    /// The token must be part of the vocabulary.
    pub fn fixed(len: usize, pad: impl Into<String>) -> Self {
        Self(Paddings::Fixed {
            len,
            pad_id: 0,
            pad_token: pad.into(),
        })
    }

    /// Validates itself.
    pub(crate) fn validate(mut self, vocab: &Vocab) -> Result<Self, Error> {
        match self.0 {
            Paddings::None => Ok(self),
            Paddings::Fixed {
                ref mut pad_id,
                ref pad_token,
                ..
            } => {
                if let Some(id) = vocab.get(pad_token) {
                    *pad_id = *id;
                    Ok(self)
                } else {
                    Err(anyhow!("padding token doesn't exist in the vocab"))
                }
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
        Encoding {
            ids: (0..len).collect(),
            ..Encoding::default()
        }
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
