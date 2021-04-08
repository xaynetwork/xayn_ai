mod analytics;
mod bert;
mod builder;
mod coi;
mod context;
mod data;
mod database;
mod error;
mod ltr;
mod mab;
mod reranker;
mod reranker_systems;
mod utils;

pub use crate::{
    builder::Builder,
    data::document::{Document, DocumentHistory, DocumentId, Relevance, UserFeedback},
    error::Error,
    reranker::{DocumentsRank, Reranker},
};

// temporary exports until the internals are wrapped
#[doc(hidden)]
pub use crate::{builder::Systems, database::InMemoryDatabaseRaw, mab::BetaSampler};

#[cfg(test)]
mod tests;
