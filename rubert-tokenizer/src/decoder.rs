/// The WordPiece decoder takes care of decoding a list of wordpiece tokens
/// back into a readable string.
pub struct Decoder {
    /// The prefix to be used for continuing subwords
    prefix: String,
    /// Whether to cleanup some tokenization artifacts (spaces before punctuation, ...)
    cleanup: bool,
}

impl Decoder {
    pub fn new(prefix: String, cleanup: bool) -> Self {
        Self { prefix, cleanup }
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            prefix: "##".into(),
            cleanup: true,
        }
    }
}

impl Decoder {
    pub(crate) fn decode(&self, tokens: Vec<&str>) -> String {
        let mut decoded = tokens
            .join(" ")
            .replace(format!(" {}", self.prefix).as_str(), "");
        if self.cleanup {
            decoded = decoded
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

        decoded
    }
}
