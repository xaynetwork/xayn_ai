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

#[repr(transparent)] // needed for FFI
#[derive(
    Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub struct CoiId(Uuid);
