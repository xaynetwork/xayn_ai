use crate::encoding::Encoding;

pub enum Padding {
    None,
    BatchLongest {
        pad_id: u32,
        pad_type_id: u32,
        pad_token: String,
        pad_to_multiple_of: Option<usize>,
    },
    Fixed {
        size: usize,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: String,
        pad_to_multiple_of: Option<usize>,
    },
}

impl Default for Padding {
    fn default() -> Self {
        Self::None
    }
}

impl Padding {
    pub fn pad_encoding(&self, encoding: Encoding) -> Encoding {
        match *self {
            Padding::None => encoding,
            Padding::BatchLongest { .. } => encoding,
            Padding::Fixed {
                mut size,
                pad_id,
                pad_type_id,
                ref pad_token,
                pad_to_multiple_of,
            } => {
                if let Some(multiple) = pad_to_multiple_of {
                    if multiple > 0 && size % multiple > 0 {
                        size += multiple - size % multiple;
                    }
                }
                encoding.pad(size, pad_id, pad_type_id, pad_token.as_str())
            }
        }
    }

    pub fn pad_encodings(&self, encodings: Vec<Encoding>) -> Vec<Encoding> {
        if encodings.is_empty() {
            return encodings;
        }

        let (mut size, pad_id, pad_type_id, pad_token, multiple) = match *self {
            Padding::None => return encodings,
            Padding::BatchLongest {
                pad_id,
                pad_type_id,
                ref pad_token,
                pad_to_multiple_of,
            } => {
                let size = encodings
                    .iter()
                    .map(|encoding| encoding.get_ids().len())
                    .max()
                    // safe unwrap: empty encodings have been returned early
                    .unwrap();
                (
                    size,
                    pad_id,
                    pad_type_id,
                    pad_token.as_str(),
                    pad_to_multiple_of.unwrap_or_default(),
                )
            }
            Padding::Fixed {
                size,
                pad_id,
                pad_type_id,
                ref pad_token,
                pad_to_multiple_of,
            } => (
                size,
                pad_id,
                pad_type_id,
                pad_token.as_str(),
                pad_to_multiple_of.unwrap_or_default(),
            ),
        };

        if multiple > 0 && size % multiple > 0 {
            size += multiple - size % multiple;
        }

        encodings
            .into_iter()
            .map(|encoding| encoding.pad(size, pad_id, pad_type_id, pad_token))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch_padding(pad_to_multiple_of: Option<usize>) -> Padding {
        Padding::BatchLongest {
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
            pad_to_multiple_of,
        }
    }

    fn fixed_padding(size: usize, pad_to_multiple_of: Option<usize>) -> Padding {
        Padding::Fixed {
            size,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".into(),
            pad_to_multiple_of,
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
    fn test_batch_padding() {
        let encodings = batch_padding(None).pad_encodings(encodings());
        assert!(encodings.iter().all(|encoding| encoding.ids.len() == 5));
    }

    #[test]
    fn test_batch_padding_multiple() {
        let encodings = batch_padding(Some(6)).pad_encodings(encodings());
        assert!(encodings.iter().all(|encoding| encoding.ids.len() == 6));
    }

    #[test]
    fn test_fixed_padding() {
        let encodings = fixed_padding(7, None).pad_encodings(encodings());
        assert!(encodings.iter().all(|encoding| encoding.ids.len() == 7));
    }

    #[test]
    fn test_fixed_padding_multiple() {
        let encodings = fixed_padding(7, Some(8)).pad_encodings(encodings());
        assert!(encodings.iter().all(|encoding| encoding.ids.len() == 8));
    }

    #[test]
    fn test_padding_zero() {
        batch_padding(Some(0)).pad_encodings(encodings());
        fixed_padding(0, Some(0)).pad_encodings(encodings());
        fixed_padding(7, Some(0)).pad_encodings(encodings());
    }
}
