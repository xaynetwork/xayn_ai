#![allow(dead_code)]

mod added_vocabulary;
mod decoder;
mod encoding;
mod model;
mod normalizer;
mod padding;
mod pipeline;
mod pre_tokenizer;
mod processor;
mod truncation;

type Error = anyhow::Error;

pub use crate::{normalizer::BertNormalizer, padding::Padding, truncation::Truncation};
