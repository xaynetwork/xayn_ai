#![allow(dead_code)]

mod builder;
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
    normalizer::Normalizer,
    padding::Padding,
    post_tokenizer::PostTokenizer,
    pre_tokenizer::PreTokenizer,
    tokenizer::Tokenizer,
    truncation::Truncation,
};
