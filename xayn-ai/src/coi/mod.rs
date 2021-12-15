mod config;
pub(crate) mod key_phrase;
mod merge;
pub(crate) mod point;
mod stats;
mod system;
mod utils;

pub(crate) use config::Configuration;
pub(crate) use merge::reduce_cois;
#[cfg(test)]
pub(crate) use system::CoiSystemError;
pub(crate) use system::{CoiSystem, NeutralCoiSystem};

use derive_more::From;
use displaydoc::Display;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[cfg(test)]
use crate::tests::mock_uuid;

#[repr(transparent)] // needed for FFI
#[derive(
    Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub struct CoiId(Uuid);

#[cfg(test)]
impl CoiId {
    /// Creates a mocked CoI id from a mocked UUID, cf. [`mock_uuid()`].
    pub(crate) const fn mocked(sub_id: usize) -> Self {
        Self(mock_uuid(sub_id))
    }
}

#[derive(Debug, Display, Error)]
pub(crate) enum CoiError {
    /// The key phrase is invalid (ie. either empty words or non-finite point)
    InvalidKeyPhrase,
    /// The key phrases are not unique
    DuplicateKeyPhrases,
}
