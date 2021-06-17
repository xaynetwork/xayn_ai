use rubert::SMBert;

use crate::{
    data::document_data::{DocumentDataWithDocument, DocumentDataWithSMBert, SMBertComponent},
    error::Error,
    reranker::systems::SMBertSystem,
};

impl SMBertSystem for SMBert {
    fn compute_embedding(
        &self,
        documents: Vec<DocumentDataWithDocument>,
    ) -> Result<Vec<DocumentDataWithSMBert>, Error> {
        // TODO: optional parallelization
        documents
            .into_iter()
            .map(|document| {
                let embedding = self.run(document.document_content.title.as_str());
                embedding
                    .map(|embedding| {
                        DocumentDataWithSMBert::from_document(
                            document,
                            SMBertComponent { embedding },
                        )
                    })
                    .map_err(Into::into)
            })
            .collect()
    }
}
