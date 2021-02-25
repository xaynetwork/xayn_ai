use crate::encoding::Encoding;

/// A padding strategy.
pub enum Padding {
    /// No padding.
    None,
    /// Padding to a fixed length.
    Fixed { len: usize, pad: String, id: u32 },
}

impl Default for Padding {
    fn default() -> Self {
        Self::None
    }
}

impl Padding {
    pub(crate) fn pad(&self, encoding: Encoding) -> Encoding {
        match self {
            Padding::None => encoding,
            Padding::Fixed { len, pad, id } => encoding.pad(*len, *id, 0, pad),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn padding(len: usize) -> Padding {
        Padding::Fixed {
            len,
            pad: "[PAD]".into(),
            id: 0,
        }
    }

    #[test]
    fn test_padding() {
        let encodings = vec![
            Encoding {
                ids: vec![0, 1, 2, 3, 4],
                ..Encoding::default()
            },
            Encoding {
                ids: vec![0, 1, 2],
                ..Encoding::default()
            },
        ];

        assert_eq!(padding(3).pad(encodings[0].clone()).len(), 5);
        assert_eq!(padding(3).pad(encodings[1].clone()).len(), 3);

        assert_eq!(padding(5).pad(encodings[0].clone()).len(), 5);
        assert_eq!(padding(5).pad(encodings[1].clone()).len(), 5);

        assert_eq!(padding(7).pad(encodings[0].clone()).len(), 7);
        assert_eq!(padding(7).pad(encodings[1].clone()).len(), 7);
    }
}
