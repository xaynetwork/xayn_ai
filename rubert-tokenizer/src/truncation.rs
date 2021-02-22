use crate::encoding::Encoding;

pub enum Truncation {
    None,
    Fixed { max_length: usize, stride: usize },
}

impl Default for Truncation {
    fn default() -> Self {
        Self::None
    }
}

impl Truncation {
    pub fn truncate_encoding(&self, encoding: Encoding, added_tokens: usize) -> Encoding {
        match *self {
            Self::None => encoding,
            Self::Fixed { max_length, stride } => {
                // TODO: fix underflow
                if max_length <= added_tokens || max_length < encoding.ids.len() + added_tokens {
                    encoding.truncate(max_length - added_tokens, stride)
                } else {
                    encoding
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truncation(max_length: usize) -> Truncation {
        Truncation::Fixed {
            max_length,
            stride: 0,
        }
    }

    fn encoding() -> Encoding {
        Encoding {
            ids: vec![1, 2, 3, 4],
            type_ids: vec![0, 0, 0, 0],
            tokens: vec![
                String::from("a"),
                String::from("b"),
                String::from("c"),
                String::from("d"),
            ],
            words: vec![Some(0), Some(1), Some(2), Some(3)],
            offsets: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
            special_tokens_mask: vec![0, 0, 0, 0],
            attention_mask: vec![1, 1, 1, 1],
            ..Encoding::default()
        }
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncation(3).truncate_encoding(encoding(), 0).ids.len(), 3);
        assert_eq!(truncation(4).truncate_encoding(encoding(), 0).ids.len(), 4);
        assert_eq!(truncation(5).truncate_encoding(encoding(), 0).ids.len(), 4);
    }

    #[test]
    fn test_truncate_zero() {
        assert_eq!(truncation(0).truncate_encoding(encoding(), 0).ids.len(), 0);
    }
}
