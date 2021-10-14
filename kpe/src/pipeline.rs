use displaydoc::Display;
use thiserror::Error;

use crate::{
    model::{bert::BertModel, classifier::ClassifierModel, cnn::CnnModel, ModelError},
    tokenizer::{key_phrase::RankedKeyPhrases, Tokenizer, TokenizerError},
};

/// A pipeline for a bert model.
///
/// Can be created via the [`Builder`] and consists of a tokenizer, a Bert model, a CNN model and a
/// Classifier model.
///
/// [`Builder`]: crate::builder::Builder
pub struct Pipeline {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) bert: BertModel,
    pub(crate) cnn: CnnModel,
    pub(crate) classifier: ClassifierModel,
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
