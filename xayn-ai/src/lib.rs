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

pub(crate) use rubert::ndarray;

pub use crate::{
    builder::Builder,
    data::document::{Document, DocumentHistory, DocumentId},
    error::Error,
    reranker::Reranker,
};

// temporary ffi exports, most of this should be abstracted away by a builder later on
pub use crate::{
    analytics::Analytics,
    coi::{CoiSystem, CoiSystemError, Configuration as CoiConfiguration},
    data::{
        document_data::{DocumentDataWithContext, DocumentDataWithLtr, DocumentDataWithMab},
        Coi,
        UserInterests,
    },
    database::Database,
    ltr::ConstLtr,
    reranker::{DocumentsRank, RerankerData},
    reranker_systems::{
        AnalyticsSystem,
        BertSystem,
        CoiSystem as CoiSystems,
        CommonSystems,
        ContextSystem,
        LtrSystem,
        MabSystem,
    },
};

#[cfg(test)]
mod tests;
