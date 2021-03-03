use crate::{
    data::document_data::{
        DocumentDataWithDocument,
        DocumentDataWithEmbedding,
        EmbeddingComponent,
    },
    error::Error,
    reranker_systems::BertSystem,
};
use rubert::RuBert;

impl BertSystem for RuBert {
    fn compute_embedding(
        &self,
        documents: Vec<DocumentDataWithDocument>,
    ) -> Result<Vec<DocumentDataWithEmbedding>, Error> {
        // TODO: optional parallelization
        documents
            .into_iter()
            .map(|document| {
                let embedding = self.run(document.document_content.snippet.as_str());
                embedding
                    .map(|embedding| {
                        DocumentDataWithEmbedding::from_document(
                            document,
                            EmbeddingComponent { embedding },
                        )
                    })
                    .map_err(Into::into)
            })
            .collect()
    }
}
