use rubert::QAMBert;

use crate::{
    data::document_data::{DocumentDataWithQAMBert, DocumentDataWithSMBert, QAMBertComponent},
    embedding_utils::l2_norm_distance,
    error::Error,
    reranker::systems::QAMBertSystem,
};

impl QAMBertSystem for QAMBert {
    fn compute_similarity(
        &self,
        documents: Vec<DocumentDataWithSMBert>,
    ) -> Result<Vec<DocumentDataWithQAMBert>, Error> {
        if let Some(document) = documents.first() {
            let query = &document.document_content.query_words;
            let query = self.run(query)?;
            documents
                .into_iter()
                .map(|document| {
                    self.run(&document.document_content.snippet)
                        .map(|embedding| {
                            let similarity = l2_norm_distance(&query, &embedding);
                            DocumentDataWithQAMBert::from_document(
                                document,
                                QAMBertComponent { similarity },
                            )
                        })
                        .map_err(Into::into)
                })
                .collect()
        } else {
            Ok(Vec::new())
        }
    }
}

/// QAMBert system to run when QAMBert is disabled
#[allow(clippy::upper_case_acronyms)]
pub struct DummyQAMBert;

impl QAMBertSystem for DummyQAMBert {
    fn compute_similarity(
        &self,
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

#[cfg(test)]
mod tests {
    use rubert::{AveragePooler, QAMBertBuilder};

    use crate::tests::documents_with_embeddings_from_snippet_and_query;

    use super::*;

    const VOCAB: &str = "../data/rubert_v0001/vocab.txt";
    const QAMBERT_MODEL: &str = "../data/rubert_v0001/qambert.onnx";

    fn qambert() -> QAMBert {
        QAMBertBuilder::from_files(VOCAB, QAMBERT_MODEL)
            .unwrap()
            .with_token_size(90)
            .unwrap()
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler)
            .build()
            .unwrap()
    }

    fn check_similarity<Q: QAMBertSystem>(system: Q, values: &[f32]) {
        let documents = documents_with_embeddings_from_snippet_and_query(
            "Europe",
            &["Football", "Continent", "Tourist guide", "Ice cream"],
        );

        let similarities: Vec<f32> = system
            .compute_similarity(documents)
            .unwrap()
            .iter()
            .map(|document| document.qambert.similarity)
            .collect();

        assert_approx_eq!(f32, similarities, values);
    }

    #[test]
    fn test_similarity() {
        check_similarity(qambert(), &[15.445126, 10.795474, 17.740929, 15.862612]);
    }

    #[test]
    fn test_similarity_dummy() {
        check_similarity(DummyQAMBert, &[0.5, 0.5, 0.5, 0.5]);
    }

    fn check_empty_documents<Q: QAMBertSystem>(system: Q) {
        let similarities = system.compute_similarity(Vec::new()).unwrap();

        assert!(similarities.is_empty());
    }

    #[test]
    fn test_empty_documents() {
        check_empty_documents(qambert());
    }

    #[test]
    fn test_empty_documents_dummy() {
        check_empty_documents(DummyQAMBert);
    }
}
