mod analytics;
mod bert;
mod coi;
mod context;
mod data;
mod database;
mod error;
mod ltr;
mod mab;
mod reranker;
mod reranker_public;
mod reranker_systems;
mod utils;

pub use crate::{
    analytics::Analytics,
    data::document::{
        Document,
        DocumentHistory,
        DocumentId,
        DocumentsRank,
        Relevance,
        UserFeedback,
    },
    error::Error,
    reranker_public::{Builder, Reranker},
};

// temporary exports until the ffi is able to take a DatabaseRaw from dart
#[doc(hidden)]
pub use crate::database::InMemoryDatabaseRaw;

#[cfg(test)]
mod tests;
