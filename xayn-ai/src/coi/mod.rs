mod config;
mod system;
pub(crate) mod utils;

pub(crate) use config::Configuration;
pub(crate) use system::CoiSystem;

#[cfg(test)]
pub(crate) use system::CoiSystemError;
