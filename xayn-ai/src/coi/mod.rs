mod config;
mod merge;
pub(crate) mod point;
mod system;
mod utils;

pub(crate) use config::Configuration;
pub(crate) use merge::reduce_cois;
pub(crate) use system::CoiSystem;
#[cfg(test)]
pub(crate) use system::CoiSystemError;

use derive_more::From;
use serde::{Deserialize, Serialize};
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
