#[macro_use]
mod utils;

mod analytics;
mod coi;
mod context;
mod data;
mod embedding_utils;
mod error;
mod ltr;
mod mab;
mod qambert;
mod reranker;
mod smbert;

pub use crate::{
    analytics::Analytics,
    data::{
        document::{
            DayOfWeek,
            Document,
            DocumentHistory,
            DocumentId,
            QueryId,
            Relevance,
            RerankingOutcomes,
            SessionId,
            UserAction,
            UserFeedback,
        },
        CoiId,
    },
    error::Error,
    reranker::public::{Builder, Reranker},
};

#[cfg(test)]
mod tests;

// Reexport for assert_approx_eq for usage in FFI crates.
#[doc(hidden)]
pub use self::utils::ApproxAssertIterHelper;
