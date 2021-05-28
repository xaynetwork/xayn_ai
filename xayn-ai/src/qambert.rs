use crate::{
    data::document_data::{DocumentDataWithQAMBert, DocumentDataWithSMBert, QAMBertComponent},
    error::Error,
    reranker::systems::QAMBertSystem,
};

#[allow(clippy::upper_case_acronyms)]
pub struct DummyQAMBert;

impl QAMBertSystem for DummyQAMBert {
    fn compute_similarity(
        &self,
        _query: &str,
        documents: Vec<DocumentDataWithSMBert>,
    ) -> Result<Vec<DocumentDataWithQAMBert>, Error> {
        Ok(documents
            .into_iter()
            .map(|document| {
                DocumentDataWithQAMBert::from_document(
                    document,
                    QAMBertComponent { similarity: 0.5 },
                )
            })
            .collect())
    }
}