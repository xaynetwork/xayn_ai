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
                // TODO: input argument to `run()` will be more flexible as soon as we have our own
                // tokenizer module
                let sentence = vec![document.document_content.snippet.as_str()];
                let embedding = self.run(sentence);
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
