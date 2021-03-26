use crate::{
    data::{document::DocumentHistory, document_data::DocumentDataWithMab},
    error::Error,
    reranker_systems,
};

pub struct Analytics;

pub(crate) struct AnalyticsSystem;

impl reranker_systems::AnalyticsSystem for AnalyticsSystem {
    fn compute_analytics(
        &self,
        _history: &[DocumentHistory],
        _documents: &[DocumentDataWithMab],
    ) -> Result<Analytics, Error> {
        Ok(Analytics)
    }
}
