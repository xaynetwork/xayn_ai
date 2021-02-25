use std::collections::HashMap;

use ndarray::{ArrayViewD, Ix1};

use crate::{
    data::{
        document::Relevance,
        document_data::{DocumentDataWithEmbedding, DocumentDataWithMab, DocumentIdentifier},
        Coi,
        UserInterests,
    },
    DocumentHistory,
    DocumentId,
};

pub fn l2_norm(x: ArrayViewD<f32>) -> f32 {
    // https://github.com/rust-ndarray/ndarray/issues/886
    let array_1d = x.into_dimensionality::<Ix1>().unwrap();
    array_1d.dot(&array_1d).sqrt()
}

// utils for `make_user_interests`
pub enum UserInterestsStatus {
    NotEnough(UserInterests),
    Ready(UserInterests),
}

pub enum UserFeedback {
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

/// Classifies the documents into positive and negative documents based on the user feedback.
pub fn classify_documents_based_on_user_feedback<D>(
    matching_documents: Vec<(&DocumentHistory, D)>,
) -> (Vec<D>, Vec<D>) {
    let mut positive_docs = Vec::<D>::new();
    let mut negative_docs = Vec::<D>::new();

    for (history_doc, doc) in matching_documents.into_iter() {
        match user_feedback(history_doc) {
            UserFeedback::Positive => positive_docs.push(doc),
            UserFeedback::Negative => negative_docs.push(doc),
        }
    }

    (positive_docs, negative_docs)
}

/// Determines the user feedback based on the user actions.
pub fn user_feedback(history: &DocumentHistory) -> UserFeedback {
    match (history.relevance, history.is_liked) {
        (Relevance::Low, false) => UserFeedback::Negative,
        _ => UserFeedback::Positive,
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
    update_alpha_or_beta(
        counts,
        cois,
        |&mut Coi { ref mut alpha, .. }: &mut Coi, adjustment: f32| *alpha *= adjustment,
    )
}

pub fn update_beta(counts: &HashMap<usize, u16>, cois: Vec<Coi>) -> Vec<Coi> {
    update_alpha_or_beta(
        counts,
        cois,
        |&mut Coi { ref mut beta, .. }: &mut Coi, adjustment: f32| *beta *= adjustment,
    )
}

/// Counts CoI Ids of the given documents.
// e.g. documents = [d_1(coi_id_1), d_2(coi_id_2), d_3(coi_id_1)]
// -> {coi_id_1: 2, coi_id_2: 1}
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
