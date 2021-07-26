mod config;
mod merge;
mod system;
mod utils;

pub(crate) use config::Configuration;
pub(crate) use merge::reduce_cois;
pub(crate) use system::CoiSystem;

#[cfg(test)]
pub(crate) use system::CoiSystemError;
