#![allow(dead_code)]

mod builder;
mod encoding;
mod model;
mod normalizer;
mod padding;
mod post_tokenizer;
mod pre_tokenizer;
mod tokenizer;
mod truncation;

type Error = anyhow::Error;

pub use crate::{
    builder::Builder,
    normalizer::{NormalizedString, Normalizer},
    padding::Padding,
    post_tokenizer::PostTokenizer,
    pre_tokenizer::{PreTokenizedString, PreTokenizer},
    tokenizer::Tokenizer,
    truncation::Truncation,
};
