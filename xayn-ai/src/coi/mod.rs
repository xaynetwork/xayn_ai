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

// Hint: We use this id new-type in FFI so repr(transparent) needs to be kept
#[repr(transparent)]
#[derive(
    Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub struct CoiId(Uuid);
