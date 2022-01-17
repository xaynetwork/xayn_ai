pub(crate) mod key_phrase;
mod merge;
pub(crate) mod point;
mod relevance;
mod stats;
mod system;
mod utils;

#[cfg(test)]
pub(crate) use self::{
    relevance::Relevances,
    system::{compute_coi, update_user_interests, CoiSystemError},
    utils::tests::create_pos_cois,
};
pub(crate) use merge::reduce_cois;
pub(crate) use system::{compute_coi_for_embedding, CoiSystem, NeutralCoiSystem};

use derive_more::From;
use displaydoc::Display;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[cfg(test)]
use crate::tests::mock_uuid;
use crate::{embedding::utils::ArcEmbedding, ranker::config::Error as ConfigError};

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
    /// A key phrase is empty
    EmptyKeyPhrase,
    /// A key phrase has non-finite embedding values: {0:#?}
    NonFiniteKeyPhrase(ArcEmbedding),
    /// A computed relevance score isn't finite.
    NonFiniteRelevance,
    /// Invalid configuration: {0}
    InvalidConfiguration(#[from] ConfigError),
}
