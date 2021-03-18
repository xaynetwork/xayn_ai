use std::collections::HashMap;

use crate::{
    data::{
        document::{Relevance, UserFeedback},
        document_data::{DocumentDataWithMab, DocumentIdentifier},
        Coi,
    },
    ndarray::Array1,
    DocumentHistory,
    DocumentId,
};

pub fn l2_norm(array: Array1<f32>) -> f32 {
    array.dot(&array).sqrt()
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

    use super::*;
    use crate::{
        data::document_data::{DocumentDataWithEmbedding, DocumentIdComponent, EmbeddingComponent},
        ndarray::{arr1, FixedInitializer},
    };

    pub fn create_cois(points: &[impl FixedInitializer<Elem = f32>]) -> Vec<Coi> {
        points
            .iter()
            .enumerate()
            .map(|(id, point)| Coi::new(id, arr1(point.as_init_slice()).into()))
            .collect()
    }

    pub fn create_data_with_embeddings(
        embeddings: &[impl FixedInitializer<Elem = f32>],
    ) -> Vec<DocumentDataWithEmbedding> {
        embeddings
            .iter()
            .enumerate()
            .map(|(id, embedding)| create_data_with_embedding(id, embedding.as_init_slice()))
            .collect()
    }

    pub fn create_data_with_embedding(id: usize, embedding: &[f32]) -> DocumentDataWithEmbedding {
        DocumentDataWithEmbedding {
            document_id: DocumentIdComponent {
                id: DocumentId(id.to_string()),
            },
            embedding: EmbeddingComponent {
                embedding: arr1(embedding).into(),
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
        let cois = create_cois(&[[1., 0., 0.], [1., 0., 0.]]);
        let counts = hashmap! {
            0 => 1,
            1 => 2,
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
        let cois = create_cois(&[[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]);

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
    #[allow(clippy::float_cmp)] // false positive, it actually compares ndarrays
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
        assert_eq!(positive_docs[0].embedding.embedding, [3., 2., 1.]);
        assert_eq!(positive_docs[1].embedding.embedding, [4., 5., 6.]);

        assert_eq!(negative_docs.len(), 1);
        assert_eq!(negative_docs[0].embedding.embedding, [1., 2., 3.]);
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
        documents.push(create_data_with_embedding(5, &[4., 5., 6.]));

        let matching_documents = collect_matching_documents(&history, &documents);

        assert_eq!(matching_documents.len(), 2);
        assert_eq!(matching_documents[0].0.id.0, "0");
        assert_eq!(matching_documents[0].1.id().0, "0");

        assert_eq!(matching_documents[1].0.id.0, "1");
        assert_eq!(matching_documents[1].1.id().0, "1");
    }
}
