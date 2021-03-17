pub mod string;

use std::{
    collections::HashMap,
    io::{BufRead, Error as IoError},
};

use displaydoc::Display;
use num_traits::FromPrimitive;
use thiserror::Error;

use crate::{
    model::string::TokenizedString,
    pre_tokenizer::string::PreTokenizedString,
    SmallString,
};

/// A vocabulary mapping tokens to ids.
pub type Vocab<N> = HashMap<String, N>;

/// A Bert word piece model.
pub struct Model<N> {
    pub vocab: Vocab<N>,
    pub unk_id: N,
    pub unk_token: SmallString,
    pub prefix: SmallString,
    pub max_chars: usize,
}

/// The potential errors of the word piece model.
#[derive(Debug, Display, Error)]
pub enum ModelError {
    /// Overflowing output data type.
    DataType,
    /// Failed to parse the vocabulary: {0}
    Vocab(#[from] IoError),
    /// Missing the unknown token in the vocabulary
    UnkToken,
    /// Missing the continuing subword prefix in the vocabulary
    SubwordPrefix,
}

impl<N> Model<N> {
    /// Parses the vocabulary.
    pub fn parse_vocab(vocab: impl BufRead) -> Result<Vocab<N>, ModelError>
    where
        N: FromPrimitive,
    {
        vocab
            .lines()
            .enumerate()
            .map(|(idx, word)| -> Result<(String, N), ModelError> {
                Ok((
                    word?.trim().to_string(),
                    N::from_usize(idx).ok_or(ModelError::DataType)?,
                ))
            })
            .collect()
    }

    /// Creates a Bert word piece model.
    pub fn new(
        vocab: Vocab<N>,
        unk_token: SmallString,
        prefix: SmallString,
        max_chars: usize,
    ) -> Result<Self, ModelError>
    where
        N: Copy,
    {
        let unk_id = vocab
            .get(unk_token.as_str())
            .copied()
            .ok_or(ModelError::UnkToken)?;

        if !vocab.keys().any(|word| word.contains(prefix.as_str())) {
            return Err(ModelError::SubwordPrefix);
        }

        Ok(Model {
            vocab,
            unk_id,
            unk_token,
            prefix,
            max_chars,
        })
    }

    /// Tokenizes the sequences.
    pub fn tokenize(&self, sequence: PreTokenizedString) -> TokenizedString<N>
    where
        N: Copy,
    {
        TokenizedString::from(sequence).tokenize(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{normalizer::string::Offsets, pre_tokenizer::PreTokenizer};

    fn assert_eq<N>(actual: TokenizedString<N>, expected: Vec<Vec<(&str, N, Offsets)>>)
    where
        N: std::fmt::Debug + PartialEq,
    {
        assert_eq!(actual.splits.len(), expected.len());
        for (split, tokens) in actual.splits.iter().zip(expected) {
            assert_eq!(split.tokens.len(), tokens.len());
            let offset = split.normalized.offset;
            for (token, (value, id, offsets)) in split.tokens.iter().zip(tokens) {
                assert_eq!(token.id, id);
                assert_eq!(token.value, value);
                assert_eq!(offset + token.offsets.0, offsets.0);
                assert_eq!(offset + token.offsets.1, offsets.1);
            }
        }
    }

    #[test]
    fn test_tokenize() {
        let vocab =
            Model::<u32>::parse_vocab(["[UNK]", "foo", "##bar"].join("\n").as_bytes()).unwrap();
        let model = Model::new(vocab, "[UNK]".into(), "##".into(), 10).unwrap();
        let pre_tokenized = PreTokenizer.pre_tokenize("foo bar foobar".into());

        let tokenized = model.tokenize(pre_tokenized);
        let expected = vec![
            vec![("foo", model.vocab["foo"], Offsets(0, 3))],
            vec![("[UNK]", model.vocab["[UNK]"], Offsets(4, 7))],
            vec![
                ("foo", model.vocab["foo"], Offsets(8, 11)),
                ("##bar", model.vocab["##bar"], Offsets(11, 14)),
            ],
        ];
        assert_eq(tokenized, expected);
    }

    #[test]
    fn test_tokenize_empty() {
        let vocab = Model::<u32>::parse_vocab(["[UNK]", "##"].join("\n").as_bytes()).unwrap();
        let model = Model::new(vocab, "[UNK]".into(), "##".into(), 10).unwrap();
        let pre_tokenized = PreTokenizer.pre_tokenize("".into());

        let tokenized = model.tokenize(pre_tokenized);
        let expected = vec![];
        assert_eq(tokenized, expected);
    }

    #[test]
    fn test_tokenize_unknown() {
        let vocab =
            Model::<u32>::parse_vocab(["[UNK]", "foo", "##bar"].join("\n").as_bytes()).unwrap();
        let model = Model::new(vocab, "[UNK]".into(), "##".into(), 10).unwrap();
        let pre_tokenized = PreTokenizer.pre_tokenize("baz bazbaz".into());

        let tokenized = model.tokenize(pre_tokenized);
        let expected = vec![
            vec![("[UNK]", model.vocab["[UNK]"], Offsets(0, 3))],
            vec![("[UNK]", model.vocab["[UNK]"], Offsets(4, 10))],
        ];
        assert_eq(tokenized, expected);
    }
}
