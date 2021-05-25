//! I/O types for reranking.

pub(crate) mod document;
pub(crate) mod history;
pub(crate) mod outcomes;

pub use self::{
    document::{CDocument, CDocuments},
    history::{CHistories, CHistory},
    outcomes::{reranking_outcomes_drop, CRerankingOutcomes},
};
