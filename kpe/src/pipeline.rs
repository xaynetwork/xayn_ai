use displaydoc::Display;
use thiserror::Error;

use crate::{
    model::{
        bert::{BertModel, BertModelError},
        classifier::{ClassifierModel, ClassifierModelError},
        cnn::{CnnModel, CnnModelError},
    },
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
    /// Failed to run the Bert model: {0}
    BertModel(#[from] BertModelError),
    /// Failed to run the CNN model: {0}
    CnnModel(#[from] CnnModelError),
    /// Failed to run the Classifier model: {0}
    ClassifierModel(#[from] ClassifierModelError),
}

impl Pipeline {
    /// Extracts the key phrases from the sequence ranked in descending order.
    pub fn run(&self, sequence: impl AsRef<str>) -> Result<RankedKeyPhrases, PipelineError> {
        // TODO: maybe add a parameter for the max key phrases to cut the returned list
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
