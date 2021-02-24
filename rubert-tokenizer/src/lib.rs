#![allow(dead_code)]

mod builder;
mod decoder;
mod encoding;
mod model;
mod normalizer;
mod padding;
mod pre_tokenizer;
mod processor;
mod sequence;
mod tokenizer;
mod truncation;

// TODO: structured error handling
type Error = anyhow::Error;

pub use crate::{
    builder::Builder,
    model::Vocab,
    normalizer::{NormalizedString, Normalizer},
    padding::Padding,
    pre_tokenizer::{PreTokenizedString, PreTokenizer},
    sequence::Sequence,
    tokenizer::Tokenizer,
    truncation::Truncation,
};
