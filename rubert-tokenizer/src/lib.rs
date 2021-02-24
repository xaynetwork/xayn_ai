#![allow(dead_code)]

mod decoder;
mod encoding;
mod model;
mod normalizer;
mod padding;
mod pre_tokenizer;
mod processor;
mod tokenizer;
mod truncation;

// TODO: structured error handling
type Error = anyhow::Error;

pub use crate::{
    normalizer::{NormalizedString, Normalizer},
    padding::Padding,
    pre_tokenizer::{PreTokenizedString, PreTokenizer},
    tokenizer::{Tokenizer, TokenizerBuilder},
    truncation::Truncation,
};
