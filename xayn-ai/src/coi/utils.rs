use std::collections::HashMap;

use crate::{
    data::document::{Relevance, UserFeedback},
    reranker::systems::CoiSystemData,
    DocumentHistory,
    DocumentId,
};

enum DocumentRelevance {
    Positive,
    Negative,
}

/// Collects all documents that are present in the history.
/// The history contains the user feedback of these documents.
pub(super) fn collect_matching_documents<'hist, 'doc>(
    history: &'hist [DocumentHistory],
    documents: &'doc [&dyn CoiSystemData],
) -> Vec<(&'hist DocumentHistory, &'doc dyn CoiSystemData)> {
    let history: HashMap<&DocumentId, &DocumentHistory> =
        history.iter().map(|dh| (&dh.id, dh)).collect();

    documents
        .iter()
        .filter_map(|doc| {
            let dh = *history.get(&doc.id())?;
            Some((dh, *doc))
        })
        .collect()
}

/// Classifies the documents into positive and negative documents based on the user feedback
/// and the relevance of the results.
pub(super) fn classify_documents_based_on_user_feedback<D>(
    matching_documents: Vec<(&DocumentHistory, D)>,
) -> (Vec<D>, Vec<D>) {
    let mut positive_docs = Vec::<D>::new();
    let mut negative_docs = Vec::<D>::new();

    for (history_doc, doc) in matching_documents.into_iter() {
        match document_relevance(history_doc) {
            DocumentRelevance::Positive => positive_docs.push(doc),
            DocumentRelevance::Negative => negative_docs.push(doc),
        }
    }

    (positive_docs, negative_docs)
}

/// Determines the [`DocumentRelevance`] based on the user feedback
/// and the relevance of the result.
fn document_relevance(history: &DocumentHistory) -> DocumentRelevance {
    match (history.relevance, history.user_feedback) {
        (Relevance::Low, UserFeedback::Irrelevant) | (Relevance::Low, UserFeedback::NotGiven) => {
            DocumentRelevance::Negative
        }
        _ => DocumentRelevance::Positive,
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::{collections::BTreeSet, time::Duration};

    use ndarray::{arr1, FixedInitializer};

    use super::*;
    use crate::{
        coi::{
            point::{CoiPoint, NegativeCoi, PositiveCoi},
            CoiId,
        },
        data::document_data::{
            CoiComponent,
            DocumentBaseComponent,
            DocumentContentComponent,
            DocumentDataWithSMBert,
            SMBertComponent,
        },
        utils::to_vec_of_ref_of,
    };

    pub(crate) struct MockCoiDoc {
        id: DocumentId,
        smbert: SMBertComponent,
        coi: Option<CoiComponent>,
    }

    impl CoiSystemData for MockCoiDoc {
        fn id(&self) -> DocumentId {
            self.id
        }

        fn smbert(&self) -> &SMBertComponent {
            &self.smbert
        }

        fn coi(&self) -> Option<&CoiComponent> {
            self.coi.as_ref()
        }

        fn viewed(&self) -> Option<Duration> {
            Some(Duration::from_secs(10))
        }
    }

    fn create_cois<FI: FixedInitializer<Elem = f32>, CP: CoiPoint>(points: &[FI]) -> Vec<CP> {
        if FI::len() == 0 {
            return Vec::new();
        }

        points
            .iter()
            .enumerate()
            .map(|(id, point)| {
                CP::new(
                    CoiId::mocked(id),
                    arr1(point.as_init_slice()).into(),
                    BTreeSet::default(),
                    Some(Duration::from_secs(10)),
                )
            })
            .collect()
    }

    pub(crate) fn create_pos_cois(
        points: &[impl FixedInitializer<Elem = f32>],
    ) -> Vec<PositiveCoi> {
        create_cois(points)
    }

    pub(crate) fn create_neg_cois(
        points: &[impl FixedInitializer<Elem = f32>],
    ) -> Vec<NegativeCoi> {
        create_cois(points)
    }

    pub(crate) fn create_data_with_embeddings(
        embeddings: &[impl FixedInitializer<Elem = f32>],
    ) -> Vec<DocumentDataWithSMBert> {
        embeddings
            .iter()
            .enumerate()
            .map(|(id, embedding)| {
                create_data_with_embedding(id as u128, id, embedding.as_init_slice())
            })
            .collect()
    }

    pub(crate) fn create_data_with_embedding(
        id: u128,
        initial_ranking: usize,
        embedding: &[f32],
    ) -> DocumentDataWithSMBert {
        DocumentDataWithSMBert {
            document_base: DocumentBaseComponent {
                id: DocumentId::from_u128(id),
                initial_ranking,
            },
            document_content: DocumentContentComponent {
                ..Default::default()
            },
            smbert: SMBertComponent {
                embedding: arr1(embedding).into(),
            },
        }
    }

    pub(crate) fn create_document_history(
        points: Vec<(Relevance, UserFeedback)>,
    ) -> Vec<DocumentHistory> {
        points
            .into_iter()
            .enumerate()
            .map(|(id, (relevance, user_feedback))| DocumentHistory {
                id: DocumentId::from_u128(id as u128),
                relevance,
                user_feedback,
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn test_user_feedback() {
        let mut history = DocumentHistory {
            id: DocumentId::from_u128(1),
            relevance: Relevance::Low,
            user_feedback: UserFeedback::Irrelevant,
            ..Default::default()
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Negative,
        ));

        history = DocumentHistory {
            relevance: Relevance::Medium,
            user_feedback: UserFeedback::Irrelevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive,
        ));

        history = DocumentHistory {
            relevance: Relevance::High,
            user_feedback: UserFeedback::Irrelevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive,
        ));

        history = DocumentHistory {
            relevance: Relevance::High,
            user_feedback: UserFeedback::Relevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive,
        ));

        history = DocumentHistory {
            relevance: Relevance::Medium,
            user_feedback: UserFeedback::Relevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive,
        ));

        history = DocumentHistory {
            relevance: Relevance::Low,
            user_feedback: UserFeedback::Relevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive,
        ));

        history = DocumentHistory {
            relevance: Relevance::High,
            user_feedback: UserFeedback::NotGiven,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive,
        ));

        history = DocumentHistory {
            relevance: Relevance::Medium,
            user_feedback: UserFeedback::NotGiven,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive,
        ));

        history = DocumentHistory {
            relevance: Relevance::Low,
            user_feedback: UserFeedback::NotGiven,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Negative,
        ));
    }

    #[test]
    fn test_classify_documents_based_on_user_feedback() {
        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Irrelevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);
        let docs = create_data_with_embeddings(&[[1., 2., 3.], [3., 2., 1.], [4., 5., 6.]]);
        let matching_documents = history.iter().zip(docs.iter()).collect();

        let (positive_docs, negative_docs) =
            classify_documents_based_on_user_feedback(matching_documents);

        assert_eq!(positive_docs.len(), 2);
        assert_eq!(positive_docs[0].smbert.embedding, arr1(&[3., 2., 1.]));
        assert_eq!(positive_docs[1].smbert.embedding, arr1(&[4., 5., 6.]));

        assert_eq!(negative_docs.len(), 1);
        assert_eq!(negative_docs[0].smbert.embedding, arr1(&[1., 2., 3.]));
    }

    #[test]
    fn test_collect_matching_documents() {
        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Irrelevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);

        let mut documents = create_data_with_embeddings(&[[1., 2., 3.], [3., 2., 1.]]);
        documents.push(create_data_with_embedding(5, 0, &[4., 5., 6.]));
        let documents = to_vec_of_ref_of!(documents, &dyn CoiSystemData);

        let matching_documents = collect_matching_documents(&history, &documents);

        assert_eq!(matching_documents.len(), 2);
        assert_eq!(matching_documents[0].0.id, DocumentId::from_u128(0));
        assert_eq!(matching_documents[0].1.id(), DocumentId::from_u128(0));

        assert_eq!(matching_documents[1].0.id, DocumentId::from_u128(1));
        assert_eq!(matching_documents[1].1.id(), DocumentId::from_u128(1));
    }
}
