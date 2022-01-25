use displaydoc::Display;
use layer::io::{BinParams, LoadingBinParamsFailed};
use thiserror::Error;

use crate::{
    configuration::Configuration,
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
    /// Failed to load binary parameters from a file: {0}
    BinParams(#[from] LoadingBinParamsFailed),
    /// Failed to build the model: {0}
    ModelBuild(#[source] ModelError),
}

impl Pipeline {
    pub fn from(config: Configuration) -> Result<Self, PipelineError> {
        let tokenizer = Tokenizer::new(
            config.vocab,
            config.accents,
            config.lowercase,
            config.token_size,
            config.key_phrase_max_count,
            config.key_phrase_min_score,
        )?;
        let bert = Bert::new(config.model, config.token_size).map_err(PipelineError::ModelBuild)?;
        let cnn = Cnn::new(BinParams::deserialize_from(config.cnn)?)?;
        let classifier = Classifier::new(BinParams::deserialize_from(config.classifier)?)
            .map_err(PipelineError::ModelBuild)?;

        Ok(Pipeline {
            tokenizer,
            bert,
            cnn,
            classifier,
        })
    }

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
    use crate::{Configuration, Pipeline};
    use std::error::Error;
    use test_utils::kpe::{bert, classifier, cnn, vocab};

    #[test]
    fn test_run_unique() -> Result<(), Box<dyn Error>> {
        let config = Configuration::from_files(vocab()?, bert()?, cnn()?, classifier()?)?
            .with_token_size(8)?
            .with_lowercase(false);

        let actual = Pipeline::from(config)?.run("A b c d e.")?;
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
        Ok(())
    }

    #[test]
    fn test_run_duplicate() -> Result<(), Box<dyn Error>> {
        let config = Configuration::from_files(vocab()?, bert()?, cnn()?, classifier()?)?
            .with_token_size(7)?
            .with_lowercase(false);

        let actual = Pipeline::from(config)?.run("A a A a A")?;
        let expected = ["a", "a a", "a a a", "a a a a", "a a a a a"];
        assert_eq!(actual.0, expected);
        Ok(())
    }
}
