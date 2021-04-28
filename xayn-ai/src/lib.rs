#[macro_use]
mod utils;

mod analytics;
mod coi;
mod context;
mod data;
mod error;
mod ltr;
mod mab;
mod reranker;
mod smbert;

pub use crate::{
    analytics::Analytics,
    data::{
        document::{
            Document,
            DocumentHistory,
            DocumentId,
            Relevance,
            RerankingOutcomes,
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
