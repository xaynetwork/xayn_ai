use ndarray::Array1;
use std::collections::HashMap;

use crate::{
    data::{
        document::{Relevance, UserFeedback},
        PositiveCoi,
    },
    reranker::systems::CoiSystemData,
    DocumentHistory,
    DocumentId,
};

pub(super) fn l2_norm(array: Array1<f32>) -> f32 {
    array.dot(&array).sqrt()
}

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
            let dh = *history.get(doc.id())?;
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
        (Relevance::Low, UserFeedback::Irrelevant) | (Relevance::Low, UserFeedback::None) => {
            DocumentRelevance::Negative
        }
        _ => DocumentRelevance::Positive,
    }
}

// utils for `update_user_interests`
fn update_alpha_or_beta<F>(
    docs: &[&dyn CoiSystemData],
    mut cois: Vec<PositiveCoi>,
    mut f: F,
) -> Vec<PositiveCoi>
where
    F: FnMut(&mut PositiveCoi, f32),
{
    let counts = count_coi_ids(docs);
    for coi in cois.iter_mut() {
        if let Some(count) = counts.get(&coi.id.0) {
            let adjustment = 1.1f32.powi(*count as i32);
            f(coi, adjustment);
        }
    }
    cois
}

pub(super) fn update_alpha(
    positive_docs: &[&dyn CoiSystemData],
    cois: Vec<PositiveCoi>,
) -> Vec<PositiveCoi> {
    update_alpha_or_beta(
        &positive_docs,
        cois,
        |PositiveCoi { ref mut alpha, .. }, adj| *alpha *= adj,
    )
}

pub(super) fn update_beta(
    negative_docs: &[&dyn CoiSystemData],
    cois: Vec<PositiveCoi>,
) -> Vec<PositiveCoi> {
    update_alpha_or_beta(
        &negative_docs,
        cois,
        |PositiveCoi { ref mut beta, .. }, adj| *beta *= adj,
    )
}

/// Counts CoI Ids of the given documents.
/// ```text
/// documents = [d_1(coi_id_1), d_2(coi_id_2), d_3(coi_id_1)]
/// count_coi_ids(documents) -> {coi_id_1: 2, coi_id_2: 1}
/// ```
fn count_coi_ids(documents: &[&dyn CoiSystemData]) -> HashMap<usize, u16> {
    documents
        .iter()
        .filter_map(|doc| doc.coi().map(|coi| coi.id))
        .fold(
            HashMap::with_capacity(documents.len()),
            |mut counts, coi_id| {
                counts
                    .entry(coi_id.0)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
                counts
            },
        )
}

#[cfg(test)]
pub(super) mod tests {
    use float_cmp::approx_eq;
    use ndarray::{arr1, FixedInitializer};

    use super::*;
    use crate::{
        data::{
            document_data::{
                CoiComponent,
                DocumentDataWithEmbedding,
                DocumentIdComponent,
                EmbeddingComponent,
                InitialRankingComponent,
            },
            CoiId,
            CoiPoint,
            NegativeCoi,
        },
        to_vec_of_ref_of,
    };

    pub(crate) struct MockCoiDoc {
        id: DocumentId,
        embedding: EmbeddingComponent,
        coi: Option<CoiComponent>,
    }

    impl CoiSystemData for MockCoiDoc {
        fn id(&self) -> &DocumentId {
            &self.id
        }

        fn embedding(&self) -> &EmbeddingComponent {
            &self.embedding
        }

        fn coi(&self) -> Option<&CoiComponent> {
            self.coi.as_ref()
        }
    }

    fn create_docs_from_coi_id(ids: &[usize]) -> Vec<MockCoiDoc> {
        ids.iter()
            .map(|id| MockCoiDoc {
                id: DocumentId("0".to_string()),
                embedding: EmbeddingComponent {
                    embedding: arr1(&[]).into(),
                },
                coi: Some(CoiComponent {
                    id: CoiId(*id),
                    pos_distance: 1.,
                    neg_distance: 1.,
                }),
            })
            .collect()
    }

    fn create_cois<CP: CoiPoint>(points: &[impl FixedInitializer<Elem = f32>]) -> Vec<CP> {
        points
            .iter()
            .enumerate()
            .map(|(id, point)| CP::new(id, arr1(point.as_init_slice()).into()))
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
    ) -> Vec<DocumentDataWithEmbedding> {
        embeddings
            .iter()
            .enumerate()
            .map(|(id, embedding)| create_data_with_embedding(id, id, embedding.as_init_slice()))
            .collect()
    }

    pub(crate) fn create_data_with_embedding(
        id: usize,
        initial_ranking: usize,
        embedding: &[f32],
    ) -> DocumentDataWithEmbedding {
        DocumentDataWithEmbedding {
            document_id: DocumentIdComponent {
                id: DocumentId(id.to_string()),
            },
            initial_ranking: InitialRankingComponent { initial_ranking },
            embedding: EmbeddingComponent {
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
                id: DocumentId(id.to_string()),
                relevance,
                user_feedback,
            })
            .collect()
    }

    #[test]
    fn test_update_alpha_and_beta() {
        let cois = create_cois(&[[1., 0., 0.], [1., 0., 0.]]);
        let docs = create_docs_from_coi_id(&[0, 1, 1]);
        let docs = to_vec_of_ref_of!(docs, &dyn CoiSystemData);

        let updated_cois = update_alpha(&docs, cois.clone());
        assert!(approx_eq!(f32, updated_cois[0].alpha, 1.1));
        assert!(approx_eq!(f32, updated_cois[0].beta, 1.));
        assert!(approx_eq!(f32, updated_cois[1].alpha, 1.21));
        assert!(approx_eq!(f32, updated_cois[1].beta, 1.));

        let updated_cois = update_beta(&docs, cois);
        assert!(approx_eq!(f32, updated_cois[0].alpha, 1.));
        assert!(approx_eq!(f32, updated_cois[0].beta, 1.1));
        assert!(approx_eq!(f32, updated_cois[1].alpha, 1.));
        assert!(approx_eq!(f32, updated_cois[1].beta, 1.21));
    }

    #[test]
    fn test_update_alpha_or_beta() {
        let cois = create_cois(&[[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]);
        let docs = create_docs_from_coi_id(&[0, 1, 1]);
        let docs = to_vec_of_ref_of!(docs, &dyn CoiSystemData);

        // only update the alpha of coi_id 1 and 2
        let updated_cois = update_alpha(&docs, cois.clone());

        assert!(approx_eq!(f32, updated_cois[0].alpha, 1.1));
        assert!(approx_eq!(f32, updated_cois[1].alpha, 1.21));
        assert!(approx_eq!(f32, updated_cois[2].alpha, 1.));

        // same for beta
        let updated_cois = update_beta(&docs, cois);
        assert!(approx_eq!(f32, updated_cois[0].beta, 1.1));
        assert!(approx_eq!(f32, updated_cois[1].beta, 1.21));
        assert!(approx_eq!(f32, updated_cois[2].beta, 1.));
    }

    #[test]
    fn test_update_alpha_or_beta_empty_cois() {
        let updated_cois = update_alpha(&Vec::new(), Vec::new());
        assert!(updated_cois.is_empty());

        let updated_cois = update_beta(&Vec::new(), Vec::new());
        assert!(updated_cois.is_empty());
    }

    #[test]
    fn test_user_feedback() {
        let mut history = DocumentHistory {
            id: DocumentId("1".to_string()),
            relevance: Relevance::Low,
            user_feedback: UserFeedback::Irrelevant,
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
            user_feedback: UserFeedback::None,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive,
        ));

        history = DocumentHistory {
            relevance: Relevance::Medium,
            user_feedback: UserFeedback::None,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive,
        ));

        history = DocumentHistory {
            relevance: Relevance::Low,
            user_feedback: UserFeedback::None,
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
        assert_eq!(positive_docs[0].embedding.embedding, arr1(&[3., 2., 1.]));
        assert_eq!(positive_docs[1].embedding.embedding, arr1(&[4., 5., 6.]));

        assert_eq!(negative_docs.len(), 1);
        assert_eq!(negative_docs[0].embedding.embedding, arr1(&[1., 2., 3.]));
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
        assert_eq!(matching_documents[0].0.id.0, "0");
        assert_eq!(matching_documents[0].1.id().0, "0");

        assert_eq!(matching_documents[1].0.id.0, "1");
        assert_eq!(matching_documents[1].1.id().0, "1");
    }
}
