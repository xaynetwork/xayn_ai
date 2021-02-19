use crate::{
    data::{
        document::DocumentHistory,
        document_data::{DocumentDataWithCoi, DocumentDataWithLtr, LtrComponent},
    },
    error::Error,
    reranker_systems::LtrSystem,
};

struct ConstLtr;

impl LtrSystem for ConstLtr {
    fn compute_ltr(
        &self,
        _history: &[DocumentHistory],
        documents: &[DocumentDataWithCoi],
    ) -> Result<Vec<DocumentDataWithLtr>, Error> {
        let context_value = 0.5;
        Ok(documents
            .iter()
            .cloned()
            .map(|doc| DocumentDataWithLtr::from_document(doc, LtrComponent { context_value }))
            .collect())
    }
}
