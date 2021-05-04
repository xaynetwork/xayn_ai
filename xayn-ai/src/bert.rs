use rubert::{Embedding1, SMBert};

use crate::{
    data::document_data::{
        DocumentDataWithDocument,
        DocumentDataWithEmbedding,
        EmbeddingComponent,
    },
    error::Error,
    reranker::systems::BertSystem,
};

pub(crate) type Embedding = Embedding1;

impl BertSystem for SMBert {
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
