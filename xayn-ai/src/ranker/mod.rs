pub(crate) mod public;

use kpe::Pipeline as KPE;
use rubert::SMBert;

use crate::{
    coi::{point::UserInterests, CoiSystem},
    embedding::utils::Embedding,
    error::Error,
};

/// The Ranker.
pub(crate) struct Ranker {
    #[allow(dead_code)]
    coi: CoiSystem,
    smbert: SMBert,
    #[allow(dead_code)]
    kpe: KPE,
    user_interests: UserInterests,
}

impl Ranker {
    /// Creates a new `Ranker`.
    pub(crate) fn new(
        smbert: SMBert,
        coi: CoiSystem,
        kpe: KPE,
        user_interests: UserInterests,
    ) -> Self {
        Self {
            smbert,
            coi,
            kpe,
            user_interests,
        }
    }

    /// Creates a byte representation of the internal state of the ranker.
    pub(crate) fn serialize(&self) -> Result<Vec<u8>, Error> {
        bincode::serialize(&self.user_interests).map_err(Into::into)
    }

    /// Computes the SMBert embedding of the given `sequence`.
    pub(crate) fn compute_smbert(&self, sequence: &str) -> Result<Embedding, Error> {
        self.smbert.run(sequence).map_err(Into::into)
    }
}
