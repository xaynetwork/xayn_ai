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

// temporary ffi exports, most of this will be abstracted away by a builder later on
pub use crate::{
    analytics::{Analytics, DummyAnalytics},
    coi::{CoiSystem, Configuration as CoiConfiguration},
    context::Context,
    data::document::{Relevance, UserFeedback},
    database::Database,
    ltr::ConstLtr,
    mab::{BetaSampler, MabRanking},
    reranker::RerankerData,
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
