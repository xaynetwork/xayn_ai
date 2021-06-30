#[macro_use]
mod utils;

mod analytics;
mod coi;
mod context;
mod data;
mod embedding;
mod error;
mod ltr;
mod mab;
mod reranker;

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
    reranker::{
        public::{Builder, Reranker},
        RerankMode,
    },
};

#[cfg(test)]
mod tests;

// we need to export rstest_reuse from the root for it to works.
// `use rstest_reuse` will trigger clippy::single_component_path_imports
// that is not possible to silence.
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use rstest_reuse as rstest_reuse;

// Reexport for assert_approx_eq for usage in FFI crates.
#[doc(hidden)]
pub use self::utils::ApproxAssertIterHelper;
