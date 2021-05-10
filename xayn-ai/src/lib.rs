#![forbid(unsafe_op_in_unsafe_fn)]

#[macro_use]
mod utils;

mod analytics;
mod bert;
mod coi;
mod context;
mod data;
mod error;
mod ltr;
mod mab;
mod reranker;

pub use crate::{
    analytics::Analytics,
    data::document::{Document, DocumentHistory, DocumentId, Ranks, Relevance, UserFeedback},
    error::Error,
    reranker::public::{Builder, Reranker},
};

#[cfg(test)]
mod tests;
