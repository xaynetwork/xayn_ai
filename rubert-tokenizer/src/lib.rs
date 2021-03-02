#![allow(dead_code)]

mod builder;
mod model;
mod normalizer;
mod post_tokenizer;
mod pre_tokenizer;
mod tokenizer;

type Error = anyhow::Error;

pub use crate::{
    builder::Builder,
    normalizer::Normalizer,
    post_tokenizer::{padding::Padding, truncation::Truncation, PostTokenizer},
    pre_tokenizer::PreTokenizer,
    tokenizer::Tokenizer,
};
