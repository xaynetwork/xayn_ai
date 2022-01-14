mod config;
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
};
pub(crate) use config::Configuration;
pub(crate) use merge::reduce_cois;
pub(crate) use system::{CoiSystem, NeutralCoiSystem};

use derive_more::From;
use displaydoc::Display;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::embedding::utils::ArcEmbedding;
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
#[allow(clippy::enum_variant_names)]
pub(crate) enum CoiError {
    /// A key phrase is empty
    EmptyKeyPhrase,
    /// A key phrase has non-finite embedding values: {0:#?}
    NonFiniteKeyPhrase(ArcEmbedding),
    /// A computed relevance score isn't finite.
    NonFiniteRelevance,
    /// Invalid coi shift factor, expected value from the unit interval
    InvalidShiftFactor,
    /// Invalid coi threshold, expected non-negative value
    #[cfg(test)]
    InvalidThreshold,
    /// Invalid coi neighbors, expected positive value
    InvalidNeighbors,
    /// Invalid coi gamma, expected value from the unit interval
    InvalidGamma,
    /// Invalid coi penalty, expected non-empty, finite and sorted values
    InvalidPenalty,
}
