//! I/O types for reranking.

pub(crate) mod document;
pub(crate) mod history;
pub(crate) mod rank;

pub use self::{
    document::{CDocument, CDocuments},
    history::{CFeedback, CHistories, CHistory, CRelevance},
    rank::{ranks_drop, CRanks},
};
