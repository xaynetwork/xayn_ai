use std::{
    borrow::Cow,
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
};

use anyhow::anyhow;

use crate::{normalizer::Offsets, tokenizer::Token, Error};

pub type Vocab = HashMap<String, u32>;
type VocabR = HashMap<u32, String>;

struct Config {
    files: Option<String>,
    vocab: Vocab,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

pub struct WordPieceBuilder {
    config: Config,
}

impl Default for WordPieceBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,
                vocab: HashMap::new(),
                unk_token: String::from("[UNK]"),
                continuing_subword_prefix: String::from("##"),
                max_input_chars_per_word: 100,
            },
        }
    }
}

impl WordPieceBuilder {
    /// Construct a new `WordPieceBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input files.
    pub fn files(mut self, vocab: String) -> Self {
        self.config.files = Some(vocab);
        self
    }

    /// Set the vocab (token -> ID) mapping.
    pub fn vocab(mut self, vocab: Vocab) -> Self {
        self.config.vocab = vocab;
        self
    }

    /// The the `UNK` token for the vocab.
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = unk_token;
        self
    }

    /// Set the prefix for continuing subwords.
    pub fn continuing_subword_prefix(mut self, continuing_subword_prefix: String) -> Self {
        self.config.continuing_subword_prefix = continuing_subword_prefix;
        self
    }

    /// Set the maximum number of input characters per word.
    pub fn max_input_chars_per_word(mut self, max_input_chars_per_word: usize) -> Self {
        self.config.max_input_chars_per_word = max_input_chars_per_word;
        self
    }

    /// Contructs a `WordPiece` model that uses the `WordPieceBuilder`'s configuration.
    pub fn build(mut self) -> Result<WordPiece, Error> {
        if let Some(vocab) = self.config.files {
            self.config.vocab = WordPiece::read_file(&vocab)?;
        }

        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();

        Ok(WordPiece {
            vocab: self.config.vocab,
            vocab_r,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            max_input_chars_per_word: self.config.max_input_chars_per_word,
        })
    }
}

pub struct WordPiece {
    vocab: Vocab,
    vocab_r: VocabR,
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

impl WordPiece {
    /// Read the given files to extract the vocab
    pub fn read_file(vocab: &str) -> Result<Vocab, Error> {
        let file = File::open(vocab)?;
        let file = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        Ok(vocab)
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    pub fn tokenize(&self, sequence: &str) -> Result<Vec<Token>, Error> {
        let char_len = sequence.chars().count();
        if char_len > self.max_input_chars_per_word {
            return Ok(vec![Token {
                value: self.unk_token.clone(),
                id: *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(anyhow!("MissingUnkToken"))?,
                offsets: Offsets(0, sequence.len()),
            }]);
        }

        let mut is_bad = false;
        let mut start = 0;
        let mut sub_tokens: Vec<Token> = vec![];

        while start < sequence.len() {
            let mut end = sequence.len();
            let mut cur_str = None;

            while start < end {
                let mut substr: Cow<str> = Cow::Borrowed(&sequence[start..end]);

                if start > 0 {
                    substr = Cow::Owned(format!("{}{}", self.continuing_subword_prefix, substr));
                }
                if self.vocab.contains_key(substr.as_ref()) {
                    cur_str = Some(Token {
                        id: self.vocab[substr.as_ref()],
                        value: substr.to_string(),
                        offsets: Offsets(start, end),
                    });
                    break;
                }
                end -= substr.chars().last().map_or(1, |c| c.len_utf8());
            }

            if cur_str.is_none() {
                is_bad = true;
                break;
            }

            sub_tokens.push(cur_str.unwrap());
            start = end;
        }

        if is_bad {
            Ok(vec![Token {
                value: self.unk_token.clone(),
                id: *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(anyhow!("MissingUnkToken"))?,
                offsets: Offsets(0, sequence.len()),
            }])
        } else {
            Ok(sub_tokens)
        }
    }
}
