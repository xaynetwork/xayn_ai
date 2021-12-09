use rubert::QAMBert;

use crate::{
    data::document_data::{DocumentDataWithCoi, DocumentDataWithQAMBert, QAMBertComponent},
    embedding::utils::l2_distance,
    error::Error,
    reranker::systems::QAMBertSystem,
};

#[cfg(feature = "multithreaded")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

impl QAMBertSystem for QAMBert {
    fn compute_similarity(
        &self,
        documents: &[DocumentDataWithCoi],
    ) -> Result<Vec<DocumentDataWithQAMBert>, Error> {
        if let Some(document) = documents.first() {
            let query = &document.document_content.query_words;
            let query = self.run(query)?;

            #[cfg(not(feature = "multithreaded"))]
            let documents = documents.iter();
            #[cfg(feature = "multithreaded")]
            let documents = documents.into_par_iter();

            documents
                .map(|document| {
                    let snippet = &document.document_content.snippet;
                    // not all documents have a snippet, if snippet is empty we use the title
                    let data = if snippet.is_empty() {
                        &document.document_content.title
                    } else {
                        snippet
                    };

                    self.run(&data)
                        .map(|embedding| {
                            let similarity = l2_distance(query.view(), embedding.view());
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
pub struct NeutralQAMBert;

impl NeutralQAMBert {
    const SIMILARITY: f32 = 0.5;
}

impl QAMBertSystem for NeutralQAMBert {
    fn compute_similarity(
        &self,
        documents: &[DocumentDataWithCoi],
    ) -> Result<Vec<DocumentDataWithQAMBert>, Error> {
        Ok(documents
            .iter()
            .map(|document| {
                DocumentDataWithQAMBert::from_document(
                    document,
                    QAMBertComponent {
                        similarity: Self::SIMILARITY,
                    },
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use rubert::{AveragePooler, QAMBertBuilder};
    use test_utils::{
        assert_approx_eq,
        qambert::{model, vocab},
    };

    use super::*;
    use crate::tests::documents_with_embeddings_from_snippet_and_query;

    fn qambert() -> QAMBert {
        QAMBertBuilder::from_files(vocab().unwrap(), model().unwrap())
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
            .compute_similarity(&documents)
            .unwrap()
            .iter()
            .map(|document| document.qambert.similarity)
            .collect();

        assert_approx_eq!(f32, similarities, values);
    }

    #[test]
    fn test_similarity() {
        check_similarity(qambert(), &[14.395557, 11.348355, 16.711432, 14.539247]);
    }

    #[test]
    fn test_similarity_dummy() {
        check_similarity(NeutralQAMBert, &[0.5, 0.5, 0.5, 0.5]);
    }

    fn check_empty_documents<Q: QAMBertSystem>(system: Q) {
        let similarities = system.compute_similarity(&[]).unwrap();

        assert!(similarities.is_empty());
    }

    #[test]
    fn test_empty_documents() {
        check_empty_documents(qambert());
    }

    #[test]
    fn test_empty_documents_dummy() {
        check_empty_documents(NeutralQAMBert);
    }

    #[test]
    fn use_title_if_snippet_empty() {
        // we want to check that if the snippet is empty the title is used instead.
        // to do this we check that the similarity between an empty query and a document with
        // an empty snippet is not zero.
        // `documents_with_embeddings_from_snippet_and_query` always returns a non empty title.
        let documents = documents_with_embeddings_from_snippet_and_query("", &[""]);

        let similarity = qambert().compute_similarity(&documents).unwrap()[0]
            .qambert
            .similarity;

        assert!(similarity > 1.);
    }
}
