#![allow(dead_code)]

mod decoder;
mod encoding;
mod model;
mod normalizer;
mod padding;
mod pipeline;
mod pre_tokenizer;
mod processor;
mod truncation;

// TODO: structured error handling
type Error = anyhow::Error;

pub use crate::{
    normalizer::{NormalizedString, Normalizer},
    padding::Padding,
    truncation::Truncation,
};
