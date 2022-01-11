use std::{collections::BTreeSet, ops::Deref, time::Duration};

use displaydoc::Display;
use thiserror::Error;
use uuid::Uuid;

use crate::{
    coi::{
        config::Configuration,
        point::{find_closest_coi, find_closest_coi_mut, CoiPoint, UserInterests},
        stats::CoiPointStats,
        utils::{classify_documents_based_on_user_feedback, collect_matching_documents},
        CoiId,
    },
    data::document_data::{CoiComponent, DocumentDataWithCoi, DocumentDataWithSMBert},
    embedding::utils::Embedding,
    reranker::systems::{self, CoiSystemData},
    DocumentHistory,
    Error,
};

#[derive(Error, Debug, Display)]
pub(crate) enum CoiSystemError {
    /// No CoI could be found for the given embedding
    NoCoi,
    /// No matching documents could be found
    NoMatchingDocuments,
}

pub(crate) struct CoiSystem {
    config: Configuration,
}

impl Default for CoiSystem {
    fn default() -> Self {
        Self::new(Configuration::default())
    }
}

impl CoiSystem {
    /// Creates a new centre of interest system.
    pub(crate) fn new(config: Configuration) -> Self {
        Self { config }
    }
}

impl systems::CoiSystem for CoiSystem {
    fn compute_coi(
        &self,
        documents: &[DocumentDataWithSMBert],
        user_interests: &UserInterests,
    ) -> Result<Vec<DocumentDataWithCoi>, Error> {
        compute_coi(documents, user_interests, self.config.neighbors.get())
    }

    fn update_user_interests(
        &self,
        history: &[DocumentHistory],
        documents: &[&dyn CoiSystemData],
        user_interests: UserInterests,
    ) -> Result<UserInterests, Error> {
        update_user_interests(
            history,
            documents,
            user_interests,
            self.config.neighbors.get(),
            self.config.threshold,
            self.config.shift_factor,
        )
    }
}

/// Assigns a CoI for the given embedding.
///
/// Returns `None` if no CoI could be found otherwise it returns the Id of
/// the CoL along with the positive and negative distance. The negative distance
/// will be [`f32::MAX`], if no negative coi could be found.
pub(crate) fn compute_coi_for_embedding(
    embedding: &Embedding,
    user_interests: &UserInterests,
    neighbors: usize,
) -> Option<CoiComponent> {
    let (coi, pos_distance) = find_closest_coi(embedding, &user_interests.positive, neighbors)?;
    let neg_distance = match find_closest_coi(embedding, &user_interests.negative, neighbors) {
        Some((_, dis)) => dis,
        None => f32::MAX,
    };

    Some(CoiComponent {
        id: coi.id,
        pos_distance,
        neg_distance,
    })
}

fn compute_coi(
    documents: &[DocumentDataWithSMBert],
    user_interests: &UserInterests,
    neighbors: usize,
) -> Result<Vec<DocumentDataWithCoi>, Error> {
    documents
        .iter()
        .map(|document| {
            compute_coi_for_embedding(&document.smbert.embedding, user_interests, neighbors)
                .map(|coi| DocumentDataWithCoi::from_document(document, coi))
                .ok_or_else(|| CoiSystemError::NoCoi.into())
        })
        .collect()
}

/// Creates a new CoI that is shifted towards the position of `embedding`.
fn shift_coi_point(embedding: &Embedding, coi: &Embedding, shift_factor: f32) -> Embedding {
    (coi.deref() * (1. - shift_factor) + embedding.deref() * shift_factor).into()
}

/// Updates the CoIs based on the given embedding. If the embedding is close to the nearest centroid
/// (within [`Configuration.threshold`]), the centroid's position gets updated,
/// otherwise a new centroid is created.
fn update_coi<CP: CoiPoint + CoiPointStats>(
    embedding: &Embedding,
    viewed: Duration,
    mut cois: Vec<CP>,
    neighbors: usize,
    threshold: f32,
    shift_factor: f32,
) -> Vec<CP> {
    match find_closest_coi_mut(embedding, &mut cois, neighbors) {
        Some((coi, distance)) if distance < threshold => {
            coi.set_point(shift_coi_point(embedding, coi.point(), shift_factor));
            coi.set_id(Uuid::new_v4().into());
            // TODO: update key phrases
            coi.update_stats(viewed);
        }
        _ => cois.push(CP::new(
            Uuid::new_v4().into(),
            embedding.clone(),
            BTreeSet::default(), // TODO: set key phrases
            viewed,
        )),
    }
    cois
}

/// Updates the CoIs based on the embeddings of docs.
fn update_cois<CP: CoiPoint + CoiPointStats>(
    docs: &[&dyn CoiSystemData],
    cois: Vec<CP>,
    neighbors: usize,
    threshold: f32,
    shift_factor: f32,
) -> Vec<CP> {
    docs.iter().fold(cois, |cois, doc| {
        update_coi(
            &doc.smbert().embedding,
            doc.viewed(),
            cois,
            neighbors,
            threshold,
            shift_factor,
        )
    })
}

fn update_user_interests(
    history: &[DocumentHistory],
    documents: &[&dyn CoiSystemData],
    mut user_interests: UserInterests,
    neighbors: usize,
    threshold: f32,
    shift_factor: f32,
) -> Result<UserInterests, Error> {
    let matching_documents = collect_matching_documents(history, documents);

    if matching_documents.is_empty() {
        return Err(CoiSystemError::NoMatchingDocuments.into());
    }

    let (positive_docs, negative_docs) =
        classify_documents_based_on_user_feedback(matching_documents);

    user_interests.positive = update_cois(
        &positive_docs,
        user_interests.positive,
        neighbors,
        threshold,
        shift_factor,
    );
    user_interests.negative = update_cois(
        &negative_docs,
        user_interests.negative,
        neighbors,
        threshold,
        shift_factor,
    );

    Ok(user_interests)
}

/// Coi system to run when Coi is disabled
pub struct NeutralCoiSystem;

impl NeutralCoiSystem {
    pub(crate) const COI: CoiComponent = CoiComponent {
        id: CoiId(Uuid::nil()),
        pos_distance: 0.,
        neg_distance: 0.,
    };
}

impl systems::CoiSystem for NeutralCoiSystem {
    fn compute_coi(
        &self,
        documents: &[DocumentDataWithSMBert],
        _user_interests: &UserInterests,
    ) -> Result<Vec<DocumentDataWithCoi>, Error> {
        Ok(documents
            .iter()
            .map(|document| DocumentDataWithCoi::from_document(document, Self::COI))
            .collect())
    }

    fn update_user_interests(
        &self,
        _history: &[DocumentHistory],
        _documents: &[&dyn CoiSystemData],
        _user_interests: UserInterests,
    ) -> Result<UserInterests, Error> {
        unreachable!(/* should never be called on this system */)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, FixedInitializer};
    use std::f32::{consts::SQRT_2, NAN};

    use super::*;
    use crate::{
        coi::{
            point::find_closest_coi_index,
            utils::tests::{
                create_data_with_embeddings,
                create_document_history,
                create_neg_cois,
                create_pos_cois,
            },
            CoiId,
        },
        data::{
            document::{DocumentId, Relevance, UserFeedback},
            document_data::{
                ContextComponent,
                DocumentBaseComponent,
                DocumentContentComponent,
                DocumentDataWithRank,
                LtrComponent,
                QAMBertComponent,
                RankComponent,
                SMBertComponent,
            },
        },
        utils::to_vec_of_ref_of,
    };
    use test_utils::assert_approx_eq;

    pub(crate) fn create_data_with_rank(
        embeddings: &[impl FixedInitializer<Elem = f32>],
    ) -> Vec<DocumentDataWithRank> {
        embeddings
            .iter()
            .enumerate()
            .map(|(id, embedding)| DocumentDataWithRank {
                document_base: DocumentBaseComponent {
                    id: DocumentId::from_u128(id as u128),
                    initial_ranking: id,
                },
                document_content: DocumentContentComponent {
                    title: id.to_string(),
                    ..DocumentContentComponent::default()
                },
                smbert: SMBertComponent {
                    embedding: arr1(embedding.as_init_slice()).into(),
                },
                qambert: QAMBertComponent { similarity: 0.5 },
                coi: CoiComponent {
                    id: CoiId::mocked(1),
                    pos_distance: 0.1,
                    neg_distance: 0.1,
                },
                ltr: LtrComponent { ltr_score: 0.5 },
                context: ContextComponent { context_value: 0.5 },
                rank: RankComponent { rank: 0 },
            })
            .collect()
    }

    #[test]
    fn test_update_coi_add_point() {
        let mut cois = create_pos_cois(&[[30., 0., 0.], [0., 20., 0.], [0., 0., 40.]]);
        let embedding = arr1(&[1., 1., 1.]).into();
        let viewed = Duration::from_secs(10);

        let threshold = 12.;
        let (index, distance) = find_closest_coi_index(&embedding, &cois, 4).unwrap();

        assert_eq!(index, 1);
        assert_approx_eq!(f32, distance, 26.747852);
        assert!(threshold < distance);

        cois = update_coi(&embedding, viewed, cois, 4, threshold, 0.1);
        assert_eq!(cois.len(), 4);
    }

    #[test]
    fn test_update_coi_update_point() {
        let cois = create_pos_cois(&[[1., 1., 1.], [10., 10., 10.], [20., 20., 20.]]);
        let embedding = arr1(&[2., 3., 4.]).into();
        let viewed = Duration::from_secs(10);

        let cois = update_coi(&embedding, viewed, cois, 4, 12., 0.1);

        assert_eq!(cois.len(), 3);
        assert_eq!(cois[0].point, arr1(&[1.1, 1.2, 1.3]));
        assert_eq!(cois[1].point, arr1(&[10., 10., 10.]));
        assert_eq!(cois[2].point, arr1(&[20., 20., 20.]));
    }

    #[test]
    fn test_shift_coi_point() {
        let coi_point = arr1(&[1., 1., 1.]).into();
        let embedding = arr1(&[2., 3., 4.]).into();

        let updated_coi = shift_coi_point(&embedding, &coi_point, 0.1);

        assert_eq!(updated_coi, arr1(&[1.1, 1.2, 1.3]));
    }

    #[test]
    fn test_update_coi_threshold_exclusive() {
        let cois = create_pos_cois(&[[0., 0., 0.]]);
        let embedding = arr1(&[0., 0., 12.]).into();
        let viewed = Duration::from_secs(10);

        let cois = update_coi(&embedding, viewed, cois, 4, 12., 0.1);

        assert_eq!(cois.len(), 2);
        assert_eq!(cois[0].point, arr1(&[0., 0., 0.]));
        assert_eq!(cois[1].point, arr1(&[0., 0., 12.]));
    }

    #[test]
    fn test_update_cois_update_the_same_point_twice() {
        // checks that an updated coi is used in the next iteration
        let cois = create_pos_cois(&[[0., 0., 0.]]);
        let documents = create_data_with_rank(&[[0., 0., 4.9], [0., 0., 5.]]);
        let documents = to_vec_of_ref_of!(documents, &dyn CoiSystemData);

        let cois = update_cois(documents.as_slice(), cois, 4, 5., 0.1);

        assert_eq!(cois.len(), 1);
        // updated coi after first embedding = [0., 0., 0.49]
        // updated coi after second embedding = [0., 0., 0.941]
        assert_eq!(cois[0].point, arr1(&[0., 0., 0.941]));
    }

    #[test]
    fn test_compute_coi_for_embedding() {
        let positive = create_pos_cois(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        let negative = create_neg_cois(&[[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]]);
        let user_interests = UserInterests { positive, negative };
        let embedding = arr1(&[2., 3., 4.]).into();

        let coi_comp = compute_coi_for_embedding(&embedding, &user_interests, 4).unwrap();

        assert_eq!(coi_comp.id, CoiId::mocked(2));
        assert_approx_eq!(f32, coi_comp.pos_distance, 4.8904557);
        assert_approx_eq!(f32, coi_comp.neg_distance, 8.1273575);
    }

    #[test]
    fn test_compute_coi_for_embedding_empty_negative_cois() {
        let positive_cois = create_pos_cois(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        let user_interests = UserInterests {
            positive: positive_cois,
            negative: Vec::new(),
        };
        let embedding = arr1(&[2., 3., 4.]).into();

        let coi_comp = compute_coi_for_embedding(&embedding, &user_interests, 4).unwrap();

        assert_eq!(coi_comp.id, CoiId::mocked(2));
        assert_approx_eq!(f32, coi_comp.pos_distance, 4.8904557);
        assert_approx_eq!(f32, coi_comp.neg_distance, f32::MAX, ulps = 0);
    }

    #[test]
    fn test_compute_coi() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(&[[1., 4., 4.], [3., 6., 6.]]);

        let documents_coi = compute_coi(&documents, &user_interests, 4).unwrap();

        assert_eq!(documents_coi[0].coi.id, CoiId::mocked(1));
        assert_approx_eq!(f32, documents_coi[0].coi.pos_distance, 2.8996046);
        assert_approx_eq!(f32, documents_coi[0].coi.neg_distance, 3.7416575);

        assert_eq!(documents_coi[1].coi.id, CoiId::mocked(1));
        assert_approx_eq!(f32, documents_coi[1].coi.pos_distance, 5.8501925);
        assert_approx_eq!(f32, documents_coi[1].coi.neg_distance, SQRT_2);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only")]
    fn test_compute_coi_all_nan() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(&[[NAN, NAN, NAN]]);
        let _ = compute_coi(&documents, &user_interests, 4);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only")]
    fn test_compute_coi_single_nan() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(&[[1., NAN, 2.]]);
        let _ = compute_coi(&documents, &user_interests, 4);
    }

    #[test]
    fn test_update_user_interests() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);

        let user_interests = UserInterests { positive, negative };

        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Irrelevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);
        let documents = create_data_with_rank(&[[1., 4., 4.], [3., 6., 6.], [1., 1., 1.]]);
        let documents = to_vec_of_ref_of!(documents, &dyn CoiSystemData);

        let UserInterests { positive, negative } =
            update_user_interests(&history, &documents, user_interests, 4, 5., 0.1).unwrap();

        assert_eq!(positive.len(), 3);
        assert_eq!(positive[0].point, arr1(&[2.7999997, 1.9, 1.]));
        assert_eq!(positive[1].point, arr1(&[1., 2., 3.]));
        assert_eq!(positive[2].point, arr1(&[3., 6., 6.]));

        assert_eq!(negative.len(), 1);
        assert_eq!(negative[0].point, arr1(&[3.6999998, 4.9, 5.7999997]));
    }

    #[test]
    fn test_update_user_interests_no_matches() {
        let error = update_user_interests(
            &Vec::new(),
            &Vec::new(),
            UserInterests::default(),
            4,
            12.,
            0.1,
        )
        .err()
        .unwrap();
        let error = error.downcast::<CoiSystemError>().unwrap();

        assert!(matches!(error, CoiSystemError::NoMatchingDocuments));
    }
}
