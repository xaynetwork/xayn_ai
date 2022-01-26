use std::sync::Arc;

use kpe::Configuration as KpeConfiguration;
use rubert::{AveragePooler, SMBertConfig};

use crate::{
    coi::{key_phrase::KeyPhrase, point::UserInterests, CoiSystem},
    embedding::{smbert::SMBert, utils::Embedding},
    error::Error,
    ranker::{config::Configuration, document::Document},
};

pub struct Ranker(super::Ranker);

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

    /// Selects the top key phrases from the positive cois, sorted in descending relevance.
    pub fn select_top_key_phrases(&mut self, top: usize) -> Vec<KeyPhrase> {
        self.0.select_top_key_phrases(top)
    }
}

pub struct Builder<'a, P> {
    smbert_config: SMBertConfig<'a, P>,
    kpe_config: KpeConfiguration<'a>,
    user_interests: UserInterests,
}

impl<'a, P> Builder<'a, P> {
    pub fn from(smbert: SMBertConfig<'a, P>, kpe: KpeConfiguration<'a>) -> Self {
        Builder {
            smbert_config: smbert,
            kpe_config: kpe,
            user_interests: UserInterests::default(),
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

    /// Creates a `Reranker`.
    ///
    /// # Errors
    ///
    /// Fails if the SMBert or KPE cannot be initialized. For example because
    /// reading from a file failed or the bytes read are have an unexpected format.
    pub fn build(self) -> Result<Ranker, Error> {
        // TODO: (lj): Overriding passed-in configuration here seems somewhat surprising.
        //             It would make sense to pull these default out of the builder.
        let smbert_config = self
            .smbert_config
            .with_token_size(52)?
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler);

        let smbert_pipeline = rubert::Pipeline::from(smbert_config)?;
        let smbert = SMBert::from(Arc::new(smbert_pipeline));

        let kpe_config = self
            .kpe_config
            .with_token_size(150)?
            .with_accents(false)
            .with_lowercase(false);

        let kpe = kpe::Pipeline::from(kpe_config)?;

        let config = Configuration::default();
        let coi = CoiSystem::new(config.clone(), smbert.clone());
        Ok(Ranker(super::Ranker::new(
            config,
            smbert,
            coi,
            kpe,
            self.user_interests,
        )))
    }
}
