use std::{sync::Arc, time::Duration};

use kpe::Configuration as KpeConfiguration;
use rubert::{AveragePooler, SMBertConfig};

use crate::{
    coi::{key_phrase::KeyPhrase, point::UserInterests, CoiSystem},
    embedding::{smbert::SMBert, utils::Embedding},
    error::Error,
    ranker::{config::Config, document::Document},
    UserFeedback,
};

pub struct Ranker(super::system::Ranker);

impl Ranker {
    /// Creates a byte representation of the internal state of the ranker.
    pub fn serialize(&self) -> Result<Vec<u8>, Error> {
        self.0.serialize()
    }

    /// Computes the SMBert embedding of the given `sequence`.
    pub fn compute_smbert(&self, sequence: &str) -> Result<Embedding, Error> {
        self.0.compute_smbert(sequence)
    }

    /// Ranks the given documents based on the learned user interests.
    ///
    /// # Errors
    ///
    /// Fails if the scores of the documents cannot be computed.
    pub fn rank(&mut self, items: &mut [impl Document]) -> Result<(), Error> {
        self.0.rank(items)
    }

    /// Logs the document view time and updates the user interests based on the given information.
    pub fn log_document_view_time(
        &mut self,
        user_feedback: UserFeedback,
        embedding: &Embedding,
        viewed: Duration,
    ) {
        self.0
            .log_document_view_time(user_feedback, embedding, viewed)
    }

    /// Logs the user reaction and updates the user interests based on the given information.
    pub fn log_user_reaction(
        &mut self,
        user_feedback: UserFeedback,
        snippet: &str,
        embedding: &Embedding,
    ) {
        self.0.log_user_reaction(user_feedback, snippet, embedding)
    }

    /// Selects the top key phrases from the positive cois, sorted in descending relevance.
    pub fn select_top_key_phrases(&mut self, top: usize) -> Vec<KeyPhrase> {
        self.0.select_top_key_phrases(top)
    }
}

pub struct Builder<'a, P> {
    smbert_config: SMBertConfig<'a, P>,
    kpe_config: KpeConfiguration<'a>,
    user_interests: UserInterests,
    ranker_config: Config,
}

impl<'a> Builder<'a, AveragePooler> {
    pub fn from(smbert: SMBertConfig<'a, AveragePooler>, kpe: KpeConfiguration<'a>) -> Self {
        Builder {
            smbert_config: smbert,
            kpe_config: kpe,
            user_interests: UserInterests::default(),
            ranker_config: Config::default(),
        }
    }

    /// Sets the serialized state to use.
    ///
    /// # Errors
    ///
    /// Fails if the state cannot be deserialized.
    pub fn with_serialized_state(mut self, bytes: impl AsRef<[u8]>) -> Result<Self, Error> {
        self.user_interests = bincode::deserialize(bytes.as_ref())?;
        Ok(self)
    }

    /// Sets the ranker [`Config`] to use.
    pub fn with_ranker_config(mut self, config: Config) -> Self {
        self.ranker_config = config;
        self
    }

    /// Creates a [`Ranker`].
    ///
    /// # Errors
    ///
    /// Fails if the SMBert or KPE cannot be initialized. For example because
    /// reading from a file failed or the bytes read are have an unexpected format.
    pub fn build(self) -> Result<Ranker, Error> {
        let smbert_pipeline = rubert::Pipeline::from(self.smbert_config)?;
        let smbert = SMBert::from(Arc::new(smbert_pipeline));
        let kpe = kpe::Pipeline::from(self.kpe_config)?;

        let coi = CoiSystem::new(self.ranker_config.clone(), smbert.clone());
        Ok(Ranker(super::system::Ranker::new(
            self.ranker_config,
            smbert,
            coi,
            kpe,
            self.user_interests,
        )))
    }
}
