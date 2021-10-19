pub(crate) mod config;
pub(crate) mod merge;
pub(crate) mod point;
pub(crate) mod system;
mod utils;

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
