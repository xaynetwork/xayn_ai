use crate::Error;

#[derive(Clone, Debug)]
/// The WordPiece decoder takes care of decoding a list of wordpiece tokens
/// back into a readable string.
pub struct WordPieceDecoder {
    /// The prefix to be used for continuing subwords
    pub prefix: String,
    /// Whether to cleanup some tokenization artifacts (spaces before punctuation, ...)
    pub cleanup: bool,
}

impl WordPieceDecoder {
    pub fn new(prefix: String, cleanup: bool) -> Self {
        Self { prefix, cleanup }
    }
}

impl Default for WordPieceDecoder {
    fn default() -> Self {
        Self {
            prefix: String::from("##"),
            cleanup: true,
        }
    }
}

impl WordPieceDecoder {
    pub fn decode(&self, tokens: Vec<String>) -> Result<String, Error> {
        let mut output = tokens.join(" ").replace(&format!(" {}", self.prefix), "");
        if self.cleanup {
            output = output
                .replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" do not", " don't")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re");
        }

        Ok(output)
    }
}
