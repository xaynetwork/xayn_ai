#![forbid(unsafe_op_in_unsafe_fn)]

mod analytics;
mod coi;
mod context;
mod data;
mod embedding;
mod error;
mod ltr;
mod ranker;
mod reranker;
mod utils;

pub use crate::{
    analytics::Analytics,
    coi::CoiId,
    data::document::{
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
    error::Error,
    ranker::public::{Builder as RankerBuilder, Ranker},
    reranker::{
        public::{Builder, Reranker},
        RerankMode,
    },
};

// We need to re-export these, since they encapsulate the arguments
// required for pipeline construction, and are passed to builders.
pub use rubert::{QAMBertConfig, SMBertConfig};

#[cfg(test)]
mod tests;

// we need to export rstest_reuse from the root for it to work.
// `use rstest_reuse` will trigger `clippy::single_component_path_imports`
// which is not possible to silence.
#[cfg(test)]
#[allow(unused_imports)]
#[rustfmt::skip]
pub(crate) use rstest_reuse as rstest_reuse;

// Reexport for the dev-tool
#[doc(hidden)]
pub use crate::ltr::{list_net, list_net_training_data_from_history};
