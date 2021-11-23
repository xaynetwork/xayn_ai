use displaydoc::Display;
use thiserror::Error;

use crate::{
    model::{bert::Bert, classifier::Classifier, cnn::Cnn, ModelError},
    tokenizer::{key_phrase::RankedKeyPhrases, Tokenizer, TokenizerError},
};

/// A pipeline for a KPE model.
///
/// Can be created via the [`Builder`] and consists of a tokenizer, a Bert model, a CNN model and a
/// Classifier model.
///
/// [`Builder`]: crate::builder::Builder
pub struct Pipeline {
    pub(crate) tokenizer: Tokenizer<{ Cnn::KEY_PHRASE_SIZE }>,
    pub(crate) bert: Bert,
    pub(crate) cnn: Cnn,
    pub(crate) classifier: Classifier,
}

/// The potential errors of the [`Pipeline`].
#[derive(Debug, Display, Error)]
pub enum PipelineError {
    /// Failed to run the tokenizer: {0}
    Tokenizer(#[from] TokenizerError),
    /// Failed to run the model: {0}
    Model(#[from] ModelError),
}

impl Pipeline {
    /// Extracts the key phrases from the sequence ranked in descending order.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<RankedKeyPhrases, PipelineError> {
        let (encoding, key_phrases) = self.tokenizer.encode(sequence);
        let embeddings = self.bert.run(
            encoding.token_ids,
            encoding.attention_mask,
            encoding.type_ids,
        )?;
        let features = self.cnn.run(embeddings, encoding.valid_mask)?;
        let scores = self.classifier.run(features, encoding.active_mask)?;

        Ok(key_phrases.rank(scores))
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::Builder;
    use test_utils::kpe::{bert, classifier, cnn, vocab};

    #[test]
    fn test_run_unique() {
        let actual = Builder::from_files(
            vocab().unwrap(),
            bert().unwrap(),
            cnn().unwrap(),
            classifier().unwrap(),
        )
        .unwrap()
        .with_token_size(8)
        .unwrap()
        .with_lowercase(false)
        .build()
        .unwrap()
        .run("A b c d e.")
        .unwrap();
        let expected = [
            // quantized, non-quantized
            "a",
            "b",
            "d",  // c
            "e.", // d
            "c",  // e.
            "a b c d e.",
            "a b c",
            "a b",     // a b c d
            "a b c d", // a b
            "b c d e.",
            "d e.",   // c d e.
            "c d e.", // d e.
            "b c d",
            "b c",
            "c d",
        ];
        assert_eq!(actual.0, expected);
    }

    #[test]
    fn test_run_duplicate() {
        let actual = Builder::from_files(
            vocab().unwrap(),
            bert().unwrap(),
            cnn().unwrap(),
            classifier().unwrap(),
        )
        .unwrap()
        .with_token_size(7)
        .unwrap()
        .with_lowercase(false)
        .build()
        .unwrap()
        .run("A a A a A")
        .unwrap();
        let expected = ["a", "a a", "a a a", "a a a a", "a a a a a"];
        assert_eq!(actual.0, expected);
    }
}
