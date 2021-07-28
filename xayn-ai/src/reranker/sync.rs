use crate::{
    coi::reduce_cois,
    data::{CoiPoint, UserInterests},
    error::Error,
};
use anyhow::bail;
use serde::{Deserialize, Serialize};

const CURRENT_SCHEMA_VERSION: u8 = 0;

/// Synchronizable data of the reranker.
#[cfg_attr(test, derive(Clone, PartialEq, Debug))]
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct SyncData {
    pub(crate) user_interests: UserInterests,
}

impl SyncData {
    /// Deserializes a `SyncData` from `bytes`.
    pub(crate) fn deserialize(bytes: &[u8]) -> Result<Self, Error> {
        // version encoded in first byte
        let version = bytes[0];
        if version != CURRENT_SCHEMA_VERSION {
            bail!(
                "Unsupported serialized data. Found version {} expected {}.",
                version,
                CURRENT_SCHEMA_VERSION,
            );
        }

        let data = bincode::deserialize(&bytes[1..])?;
        Ok(data)
    }

    /// Serializes a `SyncData` to a byte representation.
    pub(crate) fn serialize(&self) -> Result<Vec<u8>, Error> {
        let size = bincode::serialized_size(self)? + 1;
        let mut serialized = Vec::with_capacity(size as usize);
        // version encoded in first byte
        serialized.push(CURRENT_SCHEMA_VERSION);
        bincode::serialize_into(&mut serialized, self)?;

        Ok(serialized)
    }

    /// Synchronizes with another `SyncData`.
    pub(crate) fn synchronize(&mut self, other: SyncData) {
        let Self { user_interests } = other;
        self.user_interests.append(user_interests);

        reduce_cois(&mut self.user_interests.positive);
        reduce_cois(&mut self.user_interests.negative);
    }
}

impl UserInterests {
    /// Moves all user interests of `other` into `Self`.
    pub(crate) fn append(&mut self, mut other: Self) {
        append_cois(&mut self.positive, &mut other.positive);
        append_cois(&mut self.negative, &mut other.negative);
    }
}

/// Appends `remotes` to `locals`.
///
/// Remote CoIs with ids clashing with any of the local CoIs are removed.
fn append_cois<C: CoiPoint>(locals: &mut Vec<C>, remotes: &mut Vec<C>) {
    remotes.retain(|rem| !locals.iter().any(|loc| loc.id() == rem.id()));
    locals.append(remotes);
}
