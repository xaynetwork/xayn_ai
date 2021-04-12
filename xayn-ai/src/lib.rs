mod analytics;
mod bert;
mod coi;
mod context;
mod data;
mod error;
mod ltr;
mod mab;
mod reranker;
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
    reranker::public::{Builder, Reranker},
};

// temporary exports until the ffi is able to take a DatabaseRaw from dart
#[doc(hidden)]
pub use crate::reranker::database::InMemoryDatabaseRaw;

#[cfg(test)]
mod tests;
