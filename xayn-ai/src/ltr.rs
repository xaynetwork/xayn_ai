use crate::{
    data::{
        document::DocumentHistory,
        document_data::{DocumentDataWithCenterOfInterest, DocumentDataWithLtr, LtrComponent},
    },
    error::Error,
    reranker_systems::LtrSystem,
};

struct DummyLtr;

impl LtrSystem for DummyLtr {
    fn compute_ltr(
        &self,
        _history: &[DocumentHistory],
        documents: &[DocumentDataWithCenterOfInterest],
    ) -> Result<Vec<DocumentDataWithLtr>, Error> {
        let context_value = 0.5_f32;
        Ok(documents
            .iter()
            .cloned()
            .map(|doc| DocumentDataWithLtr::from_document(doc, LtrComponent { context_value }))
            .collect())
    }
}
