use crate::encoding::Encoding;

pub enum PaddingStrategy {
    BatchLongest,
    Fixed(usize),
}

pub struct Padding {
    pub strategy: PaddingStrategy,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
}

impl Default for Padding {
    fn default() -> Self {
        Self {
            strategy: PaddingStrategy::BatchLongest,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: String::from("[PAD]"),
        }
    }
}

impl Padding {
    pub fn pad_encoding(&self, encoding: Encoding) -> Encoding {
        let mut pad_length = match self.strategy {
            PaddingStrategy::Fixed(size) => size,
            PaddingStrategy::BatchLongest => return encoding,
        };

        if let Some(multiple) = self.pad_to_multiple_of {
            if multiple > 0 && pad_length % multiple > 0 {
                pad_length += multiple - pad_length % multiple;
            }
        }

        encoding.pad(
            pad_length,
            self.pad_id,
            self.pad_type_id,
            self.pad_token.as_str(),
        )
    }

    pub fn pad_encodings(&self, encodings: Vec<Encoding>) -> Vec<Encoding> {
        if encodings.is_empty() {
            return encodings;
        }

        let mut pad_length = match self.strategy {
            PaddingStrategy::Fixed(size) => size,
            PaddingStrategy::BatchLongest => encodings
                .iter()
                .map(|encoding| encoding.get_ids().len())
                .max()
                // safe unwrap: empty encodings have been returned early
                .unwrap(),
        };

        if let Some(multiple) = self.pad_to_multiple_of {
            if multiple > 0 && pad_length % multiple > 0 {
                pad_length += multiple - pad_length % multiple;
            }
        }

        encodings
            .into_iter()
            .map(|encoding| {
                encoding.pad(
                    pad_length,
                    self.pad_id,
                    self.pad_type_id,
                    self.pad_token.as_str(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn padding(strategy: PaddingStrategy, pad_to_multiple_of: Option<usize>) -> Padding {
        Padding {
            strategy,
            pad_to_multiple_of,
            ..Padding::default()
        }
    }

    fn encodings() -> Vec<Encoding> {
        vec![
            Encoding {
                ids: vec![0, 1, 2, 3, 4],
                ..Encoding::default()
            },
            Encoding {
                ids: vec![0, 1, 2],
                ..Encoding::default()
            },
        ]
    }

    #[test]
    fn test_pad_fixed() {
        let encodings = padding(PaddingStrategy::Fixed(7), None).pad_encodings(encodings());
        assert!(encodings.iter().all(|encoding| encoding.ids.len() == 7));
    }

    #[test]
    fn test_pad_fixed_multiple() {
        let encodings = padding(PaddingStrategy::Fixed(7), Some(8)).pad_encodings(encodings());
        assert!(encodings.iter().all(|encoding| encoding.ids.len() == 8));
    }

    #[test]
    fn test_pad_batch() {
        let encodings = padding(PaddingStrategy::BatchLongest, None).pad_encodings(encodings());
        assert!(encodings.iter().all(|encoding| encoding.ids.len() == 5));
    }

    #[test]
    fn test_pad_batch_multiple() {
        let encodings = padding(PaddingStrategy::BatchLongest, Some(6)).pad_encodings(encodings());
        assert!(encodings.iter().all(|encoding| encoding.ids.len() == 6));
    }

    #[test]
    fn test_pad_zero() {
        padding(PaddingStrategy::BatchLongest, Some(0)).pad_encodings(encodings());
    }
}
