use kpe::Pipeline as KPE;
use rubert::SMBert;

use crate::{
    coi::{point::UserInterests, CoiSystem},
    error::Error,
};

/// The Ranker.
#[allow(dead_code)]
pub(crate) struct Ranker {
    coi: CoiSystem,
    smbert: SMBert,
    kpe: KPE,
    user_interests: UserInterests,
}

#[allow(dead_code)]
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
}
