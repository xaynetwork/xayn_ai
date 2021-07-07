use crate::{data::UserInterests, error::Error};
use serde::{Deserialize, Serialize};

/// Synchronizable data of the reranker.
#[cfg_attr(test, derive(Clone, PartialEq, Debug))]
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct SyncData {
    pub(crate) user_interests: UserInterests,
}

impl SyncData {
    pub(crate) fn deserialize(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }

        Ok(bincode::deserialize(&bytes)?)
    }

    pub(crate) fn serialize(&self) -> Result<Vec<u8>, Error> {
        Ok(bincode::serialize(self)?)
    }

    pub(crate) fn merge(&mut self, _other: SyncData) {
        todo!()
    }
}
