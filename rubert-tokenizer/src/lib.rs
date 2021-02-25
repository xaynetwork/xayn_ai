#![allow(dead_code)]

mod builder;
mod decoder;
mod encoding;
mod model;
mod normalizer;
mod padding;
mod post_tokenizer;
mod pre_tokenizer;
mod sequence;
mod tokenizer;
mod truncation;

type Error = anyhow::Error;

pub use crate::{
    builder::Builder,
    decoder::Decoder,
    model::Vocab,
    normalizer::{NormalizedString, Normalizer},
    padding::Padding,
    post_tokenizer::PostTokenizer,
    pre_tokenizer::{PreTokenizedString, PreTokenizer},
    sequence::Sequence,
    tokenizer::Tokenizer,
    truncation::Truncation,
};
