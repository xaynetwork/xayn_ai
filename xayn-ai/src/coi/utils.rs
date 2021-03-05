use std::collections::HashMap;

use ndarray::Array1;

use crate::{
    data::{
        document::{Relevance, UserFeedback},
        document_data::{DocumentDataWithEmbedding, DocumentDataWithMab, DocumentIdentifier},
        Coi,
        UserInterests,
    },
    DocumentHistory,
    DocumentId,
};

pub fn l2_norm(array: Array1<f32>) -> f32 {
    array.dot(&array).sqrt()
}

// utils for `make_user_interests`
pub enum UserInterestsStatus {
    NotEnough(UserInterests),
    Ready(UserInterests),
}

pub enum DocumentRelevance {
    Positive,
    Negative,
}

/// Collects all documents that are present in the history.
/// The history contains the user feedback of these documents.
pub fn collect_matching_documents<'hist, 'doc, D: DocumentIdentifier>(
    history: &'hist [DocumentHistory],
    documents: &'doc [D],
) -> Vec<(&'hist DocumentHistory, &'doc D)> {
    let history: HashMap<&DocumentId, &DocumentHistory> =
        history.iter().map(|dh| (&dh.id, dh)).collect();

    documents
        .iter()
        .filter_map(|doc| {
            let dh = *history.get(doc.id())?;
            Some((dh, doc))
        })
        .collect()
}

/// Classifies the documents into positive and negative documents based on the user feedback
/// and the relevance of the results.
pub fn classify_documents_based_on_user_feedback<D>(
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
pub fn document_relevance(history: &DocumentHistory) -> DocumentRelevance {
    match (history.relevance, history.user_feedback) {
        (Relevance::Low, UserFeedback::Irrelevant) | (Relevance::Low, UserFeedback::None) => {
            DocumentRelevance::Negative
        }
        _ => DocumentRelevance::Positive,
    }
}

/// Extends the user interests based on the positive and negative documents.
pub fn extend_user_interests_based_on_documents(
    positive_docs: Vec<&DocumentDataWithEmbedding>,
    negative_docs: Vec<&DocumentDataWithEmbedding>,
    user_interests: UserInterests,
) -> UserInterests {
    let positive = extend_user_interests(positive_docs, user_interests.positive);
    let negative = extend_user_interests(negative_docs, user_interests.negative);

    UserInterests { positive, negative }
}

/// Extends the user interests from the document embeddings.
fn extend_user_interests(
    documents: Vec<&DocumentDataWithEmbedding>,
    mut interests: Vec<Coi>,
) -> Vec<Coi> {
    let next_coi_ids = interests.len()..;

    let cois = next_coi_ids
        .into_iter()
        .zip(documents.into_iter())
        .map(|(index, doc)| Coi::new(index, doc.embedding.embedding.clone()));

    interests.extend(cois);
    interests
}

// utils for `update_user_interests`
fn update_alpha_or_beta<F>(counts: &HashMap<usize, u16>, mut cois: Vec<Coi>, mut f: F) -> Vec<Coi>
where
    F: FnMut(&mut Coi, f32),
{
    for coi in cois.iter_mut() {
        if let Some(count) = counts.get(&coi.id.0) {
            let adjustment = 1.1f32.powi(*count as i32);
            f(coi, adjustment);
        }
    }
    cois
}

pub fn update_alpha(counts: &HashMap<usize, u16>, cois: Vec<Coi>) -> Vec<Coi> {
    update_alpha_or_beta(counts, cois, |Coi { ref mut alpha, .. }, adj| *alpha *= adj)
}

pub fn update_beta(counts: &HashMap<usize, u16>, cois: Vec<Coi>) -> Vec<Coi> {
    update_alpha_or_beta(counts, cois, |Coi { ref mut beta, .. }, adj| *beta *= adj)
}

/// Counts CoI Ids of the given documents.
/// ```text
/// documents = [d_1(coi_id_1), d_2(coi_id_2), d_3(coi_id_1)]
/// count_coi_ids(documents) -> {coi_id_1: 2, coi_id_2: 1}
/// ```
pub fn count_coi_ids(documents: &[&DocumentDataWithMab]) -> HashMap<usize, u16> {
    documents.iter().fold(
        HashMap::with_capacity(documents.len()),
        |mut counts, doc| {
            counts
                .entry(doc.coi.id.0)
                .and_modify(|count| *count += 1)
                .or_insert(1);
            counts
        },
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use float_cmp::approx_eq;
    use maplit::hashmap;
    use ndarray::{array, Array1};
    use rubert::Embeddings;

    use super::*;

    use crate::data::document_data::{DocumentIdComponent, EmbeddingComponent};

    pub fn create_cois(points: Vec<Array1<f32>>) -> Vec<Coi> {
        points
            .into_iter()
            .enumerate()
            .map(|(id, point)| Coi::new(id, create_embedding(point)))
            .collect()
    }

    pub fn create_embedding(point: Array1<f32>) -> Embeddings {
        Embeddings(point.into_shared().into_dyn())
    }

    pub fn create_data_with_embeddings(points: Vec<Array1<f32>>) -> Vec<DocumentDataWithEmbedding> {
        points
            .into_iter()
            .enumerate()
            .map(|(id, point)| create_data_with_embedding(id, point))
            .collect()
    }

    pub fn create_data_with_embedding(id: usize, point: Array1<f32>) -> DocumentDataWithEmbedding {
        DocumentDataWithEmbedding {
            document_id: DocumentIdComponent {
                id: DocumentId(id.to_string()),
            },
            embedding: EmbeddingComponent {
                embedding: create_embedding(point),
            },
        }
    }

    pub fn create_document_history(points: Vec<(Relevance, UserFeedback)>) -> Vec<DocumentHistory> {
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
        let cois = create_cois(vec![array![1., 0., 0.], array![1., 0., 0.]]);
        let counts = hashmap! {
            0 => 1,
            1 => 2
        };

        let updated_cois = update_alpha(&counts, cois.clone());
        assert!(approx_eq!(f32, updated_cois[0].alpha, 1.1));
        assert!(approx_eq!(f32, updated_cois[0].beta, 1.));
        assert!(approx_eq!(f32, updated_cois[1].alpha, 1.21));
        assert!(approx_eq!(f32, updated_cois[1].beta, 1.));

        let updated_cois = update_beta(&counts, cois);
        assert!(approx_eq!(f32, updated_cois[0].alpha, 1.));
        assert!(approx_eq!(f32, updated_cois[0].beta, 1.1));
        assert!(approx_eq!(f32, updated_cois[1].alpha, 1.));
        assert!(approx_eq!(f32, updated_cois[1].beta, 1.21));
    }

    #[test]
    fn test_update_alpha_or_beta() {
        let cois = create_cois(vec![
            array![1., 0., 0.],
            array![1., 0., 0.],
            array![1., 0., 0.],
        ]);

        let counts = hashmap! {
            0 => 1,
            1 => 2
        };

        // only update the alpha of coi_id 1 and 2
        let updated_cois = update_alpha(&counts, cois.clone());

        assert!(approx_eq!(f32, updated_cois[0].alpha, 1.1));
        assert!(approx_eq!(f32, updated_cois[1].alpha, 1.21));
        assert!(approx_eq!(f32, updated_cois[2].alpha, 1.));

        // same for beta
        let updated_cois = update_beta(&counts, cois);
        assert!(approx_eq!(f32, updated_cois[0].beta, 1.1));
        assert!(approx_eq!(f32, updated_cois[1].beta, 1.21));
        assert!(approx_eq!(f32, updated_cois[2].beta, 1.));
    }

    #[test]
    fn test_update_alpha_or_beta_empty_cois() {
        let updated_cois = update_alpha(&HashMap::new(), Vec::new());
        assert!(updated_cois.is_empty());

        let updated_cois = update_beta(&HashMap::new(), Vec::new());
        assert!(updated_cois.is_empty());
    }

    #[test]
    fn test_extend_user_interests_empty_interests() {
        let docs = create_data_with_embeddings(vec![array![1., 0., 0.]]);
        let documents = vec![&docs[0]];

        let extended = extend_user_interests(documents, Vec::new());

        assert_eq!(extended.len(), 1);
        assert_eq!(extended[0].id.0, 0);
    }

    #[test]
    fn test_extend_user_interests_empty_docs() {
        let extended = extend_user_interests(Vec::new(), Vec::new());
        assert!(extended.is_empty());
    }

    #[test]
    fn test_extend_user_interests_non_empty_interests() {
        let interests = create_cois(vec![array![1., 0., 0.]]);

        let docs = create_data_with_embeddings(vec![array![1., 2., 3.]]);
        let documents = vec![&docs[0]];

        let extended = extend_user_interests(documents, interests);

        assert_eq!(extended.len(), 2);
        assert_eq!(extended[0].id.0, 0);
        assert_eq!(extended[0].point, create_embedding(array![1., 0., 0.]));

        assert_eq!(extended[1].id.0, 1);
        assert_eq!(extended[1].point, create_embedding(array![1., 2., 3.]));
    }

    #[test]
    fn test_extend_user_interests_based_on_documents() {
        let docs = create_data_with_embeddings(vec![array![1., 2., 3.], array![3., 2., 1.]]);

        let UserInterests { positive, negative } = extend_user_interests_based_on_documents(
            vec![&docs[0]],
            vec![&docs[1]],
            UserInterests::new(),
        );

        assert_eq!(positive.len(), 1);
        assert_eq!(positive[0].id.0, 0);
        assert_eq!(positive[0].point, create_embedding(array![1., 2., 3.]));

        assert_eq!(negative.len(), 1);
        assert_eq!(negative[0].id.0, 0);
        assert_eq!(negative[0].point, create_embedding(array![3., 2., 1.]));
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
            DocumentRelevance::Negative
        ));

        history = DocumentHistory {
            relevance: Relevance::Medium,
            user_feedback: UserFeedback::Irrelevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive
        ));

        history = DocumentHistory {
            relevance: Relevance::High,
            user_feedback: UserFeedback::Irrelevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive
        ));

        history = DocumentHistory {
            relevance: Relevance::High,
            user_feedback: UserFeedback::Relevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive
        ));

        history = DocumentHistory {
            relevance: Relevance::Medium,
            user_feedback: UserFeedback::Relevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive
        ));

        history = DocumentHistory {
            relevance: Relevance::Low,
            user_feedback: UserFeedback::Relevant,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive
        ));

        history = DocumentHistory {
            relevance: Relevance::High,
            user_feedback: UserFeedback::None,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive
        ));

        history = DocumentHistory {
            relevance: Relevance::Medium,
            user_feedback: UserFeedback::None,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Positive
        ));

        history = DocumentHistory {
            relevance: Relevance::Low,
            user_feedback: UserFeedback::None,
            ..history
        };
        assert!(matches!(
            document_relevance(&history),
            DocumentRelevance::Negative
        ));
    }

    #[test]
    fn test_classify_documents_based_on_user_feedback() {
        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Irrelevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);
        let docs = create_data_with_embeddings(vec![
            array![1., 2., 3.],
            array![3., 2., 1.],
            array![4., 5., 6.],
        ]);
        let matching_documents = vec![
            (&history[0], &docs[0]),
            (&history[1], &docs[1]),
            (&history[2], &docs[2]),
        ];

        let (positive_docs, negative_docs) =
            classify_documents_based_on_user_feedback(matching_documents);

        assert_eq!(positive_docs.len(), 2);
        assert_eq!(
            positive_docs[0].embedding.embedding,
            create_embedding(array![3., 2., 1.])
        );
        assert_eq!(
            positive_docs[1].embedding.embedding,
            create_embedding(array![4., 5., 6.])
        );

        assert_eq!(negative_docs.len(), 1);
        assert_eq!(
            negative_docs[0].embedding.embedding,
            create_embedding(array![1., 2., 3.])
        );
    }

    #[test]
    fn test_collect_matching_documents() {
        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Irrelevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);

        let mut documents =
            create_data_with_embeddings(vec![array![1., 2., 3.], array![3., 2., 1.]]);
        documents.push(create_data_with_embedding(5, array![4., 5., 6.]));

        let matching_documents = collect_matching_documents(&history, &documents);

        assert_eq!(matching_documents.len(), 2);
        assert_eq!(matching_documents[0].0.id.0, "0");
        assert_eq!(matching_documents[0].1.id().0, "0");

        assert_eq!(matching_documents[1].0.id.0, "1");
        assert_eq!(matching_documents[1].1.id().0, "1");
    }
}
