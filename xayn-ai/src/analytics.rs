use crate::{
    data::{document::DocumentHistory, document_data::DocumentDataWithMab},
    error::Error,
    reranker_systems::AnalyticsSystem,
};

pub struct Analytics;

pub struct DummyAnalytics;

impl AnalyticsSystem for DummyAnalytics {
    fn compute_analytics(
        &self,
        _history: &[DocumentHistory],
        _documents: &[DocumentDataWithMab],
    ) -> Result<Analytics, Error> {
        Ok(Analytics)
    }
}
