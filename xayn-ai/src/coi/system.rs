use std::ops::Deref;

use displaydoc::Display;
use thiserror::Error;
use uuid::Uuid;

use crate::{
    coi::{
        config::Configuration,
        utils::{
            classify_documents_based_on_user_feedback,
            collect_matching_documents,
            update_alpha,
            update_beta,
        },
    },
    data::{
        document_data::{CoiComponent, DocumentDataWithCoi, DocumentDataWithQAMBert},
        CoiPoint,
        UserInterests,
    },
    embedding::utils::{l2_distance, Embedding},
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

    /// Finds the closest centre of interest (CoI) for the given embedding.
    ///
    /// Returns the index of the CoI along with the weighted distance between the given embedding
    /// and the k nearest CoIs. If no CoIs were given, `None` will be returned.
    fn find_closest_coi_index(
        &self,
        embedding: &Embedding,
        cois: &[impl CoiPoint],
    ) -> Option<(usize, f32)> {
        if cois.is_empty() {
            return None;
        }

        let mut distances = cois
            .iter()
            .map(|coi| l2_distance(embedding, coi.point()))
            .enumerate()
            .collect::<Vec<_>>();
        distances.sort_by(|(_, this), (_, other)| this.partial_cmp(other).unwrap());
        let index = distances[0].0;

        let total = distances.iter().map(|(_, distance)| *distance).sum::<f32>();
        let distance = if total > 0.0 {
            distances
                .iter()
                .take(self.config.neighbors.get())
                .zip(distances.iter().take(self.config.neighbors.get()).rev())
                .map(|((_, distance), (_, reversed))| distance * (reversed / total))
                .sum()
        } else {
            0.0
        };

        Some((index, distance))
    }

    /// Finds the closest CoI for the given embedding.
    ///
    /// Returns an immutable reference to the CoI along with the weighted distance between the given
    /// embedding and the k nearest CoIs. If no CoIs were given, `None` will be returned.
    fn find_closest_coi<'coi, CP: CoiPoint>(
        &self,
        embedding: &Embedding,
        cois: &'coi [CP],
    ) -> Option<(&'coi CP, f32)> {
        let (index, distance) = self.find_closest_coi_index(embedding, cois)?;
        Some((&cois[index], distance))
    }

    /// Finds the closest CoI for the given embedding.
    ///
    /// Returns a mutable reference to the CoI along with the weighted distance between the given
    /// embedding and the k nearest CoIs. If no CoIs were given, `None` will be returned.
    fn find_closest_coi_mut<'coi, CP: CoiPoint>(
        &self,
        embedding: &Embedding,
        cois: &'coi mut [CP],
    ) -> Option<(&'coi mut CP, f32)> {
        let (index, distance) = self.find_closest_coi_index(embedding, cois)?;
        Some((&mut cois[index], distance))
    }

    /// Creates a new CoI that is shifted towards the position of `embedding`.
    fn shift_coi_point(&self, embedding: &Embedding, coi: &Embedding) -> Embedding {
        let updated = coi.deref() * (1. - self.config.shift_factor)
            + embedding.deref() * self.config.shift_factor;
        updated.into()
    }

    /// Updates the CoIs based on the given embedding. If the embedding is close to the nearest centroid
    /// (within [`Configuration.threshold`]), the centroid's position gets updated,
    /// otherwise a new centroid is created.
    fn update_coi<CP: CoiPoint>(&self, embedding: &Embedding, mut cois: Vec<CP>) -> Vec<CP> {
        match self.find_closest_coi_mut(embedding, &mut cois) {
            Some((coi, distance)) if distance < self.config.threshold => {
                coi.set_point(self.shift_coi_point(embedding, coi.point()));
                coi.set_id(Uuid::new_v4().into());
            }
            _ => cois.push(CP::new(Uuid::new_v4().into(), embedding.clone())),
        }
        cois
    }

    /// Updates the CoIs based on the embeddings of docs.
    fn update_cois<CP: CoiPoint>(&self, docs: &[&dyn CoiSystemData], cois: Vec<CP>) -> Vec<CP> {
        docs.iter().fold(cois, |cois, doc| {
            self.update_coi(&doc.smbert().embedding, cois)
        })
    }

    /// Assigns a CoI for the given embedding.
    /// Returns `None` if no CoI could be found otherwise it returns the Id of
    /// the CoL along with the positive and negative distance. The negative distance
    /// will be [`f32::MAX`], if no negative coi could be found.
    fn compute_coi_for_embedding(
        &self,
        embedding: &Embedding,
        user_interests: &UserInterests,
    ) -> Option<CoiComponent> {
        let (coi, pos_distance) = self.find_closest_coi(embedding, &user_interests.positive)?;
        let neg_distance = match self.find_closest_coi(embedding, &user_interests.negative) {
            Some((_, dis)) => dis,
            None => f32::MAX,
        };

        Some(CoiComponent {
            id: coi.id,
            pos_distance,
            neg_distance,
        })
    }
}

impl systems::CoiSystem for CoiSystem {
    fn compute_coi(
        &self,
        documents: Vec<DocumentDataWithQAMBert>,
        user_interests: &UserInterests,
    ) -> Result<Vec<DocumentDataWithCoi>, Error> {
        documents
            .into_iter()
            .map(|document| {
                let coi = self
                    .compute_coi_for_embedding(&document.smbert.embedding, user_interests)
                    .ok_or(CoiSystemError::NoCoi)?;
                Ok(DocumentDataWithCoi::from_document(document, coi))
            })
            .collect()
    }

    fn update_user_interests(
        &self,
        history: &[DocumentHistory],
        documents: &[&dyn CoiSystemData],
        mut user_interests: UserInterests,
    ) -> Result<UserInterests, Error> {
        let matching_documents = collect_matching_documents(history, documents);

        if matching_documents.is_empty() {
            return Err(CoiSystemError::NoMatchingDocuments.into());
        }

        let (positive_docs, negative_docs) =
            classify_documents_based_on_user_feedback(matching_documents);

        user_interests.positive = self.update_cois(&positive_docs, user_interests.positive);
        user_interests.negative = self.update_cois(&negative_docs, user_interests.negative);

        user_interests.positive = update_alpha(&positive_docs, user_interests.positive);
        user_interests.positive = update_beta(&negative_docs, user_interests.positive);

        Ok(user_interests)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, FixedInitializer};
    use std::f32::{consts::SQRT_2, NAN};

    use super::*;
    use crate::{
        coi::utils::tests::{
            create_data_with_embeddings,
            create_document_history,
            create_neg_cois,
            create_pos_cois,
        },
        data::{
            document::{DocumentId, Relevance, UserFeedback},
            document_data::{
                ContextComponent,
                DocumentBaseComponent,
                DocumentDataWithMab,
                LtrComponent,
                MabComponent,
                QAMBertComponent,
                SMBertComponent,
            },
            PositiveCoi,
        },
        reranker::systems::CoiSystem as CoiSystemTrait,
        to_vec_of_ref_of,
        utils::mock_coi_id,
    };

    pub(crate) fn create_data_with_mab(
        embeddings: &[impl FixedInitializer<Elem = f32>],
    ) -> Vec<DocumentDataWithMab> {
        embeddings
            .iter()
            .enumerate()
            .map(|(id, embedding)| DocumentDataWithMab {
                document_base: DocumentBaseComponent {
                    id: DocumentId::from_u128(id as u128),
                    initial_ranking: id,
                },
                smbert: SMBertComponent {
                    embedding: arr1(embedding.as_init_slice()).into(),
                },
                qambert: QAMBertComponent { similarity: 0.5 },
                coi: CoiComponent {
                    id: mock_coi_id(1),
                    pos_distance: 0.1,
                    neg_distance: 0.1,
                },
                ltr: LtrComponent { ltr_score: 0.5 },
                context: ContextComponent { context_value: 0.5 },
                mab: MabComponent { rank: 0 },
            })
            .collect()
    }

    #[test]
    fn test_find_closest_coi_index() {
        let cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let embedding = arr1(&[1., 5., 9.]).into();

        let (index, distance) = CoiSystem::default()
            .find_closest_coi_index(&embedding, &cois)
            .unwrap();

        assert_eq!(index, 1);
        assert_approx_eq!(f32, distance, 5.7716017);
    }

    #[test]
    fn test_find_closest_coi_index_equal() {
        let cois = create_pos_cois(&[[1., 2., 3.]]);
        let embedding = arr1(&[1., 2., 3.]).into();

        let (index, distance) = CoiSystem::default()
            .find_closest_coi_index(&embedding, &cois)
            .unwrap();

        assert_eq!(index, 0);
        assert_approx_eq!(f32, distance, 0.0, ulps = 0);
    }

    #[test]
    #[should_panic(expected = "vectors must consist of real values only")]
    fn test_find_closest_coi_index_all_nan() {
        let cois = create_pos_cois(&[[1., 2., 3.]]);
        let embedding = arr1(&[NAN, NAN, NAN]).into();
        CoiSystem::default().find_closest_coi_index(&embedding, &cois);
    }

    #[test]
    #[should_panic(expected = "vectors must consist of real values only")]
    fn test_find_closest_coi_index_single_nan() {
        let cois = create_pos_cois(&[[1., 2., 3.]]);
        let embedding = arr1(&[1., NAN, 2.]).into();
        CoiSystem::default().find_closest_coi_index(&embedding, &cois);
    }

    #[test]
    fn test_find_closest_coi_index_empty() {
        let embedding = arr1(&[1., 2., 3.]).into();
        let coi = CoiSystem::default().find_closest_coi_index(&embedding, &[] as &[PositiveCoi]);
        assert!(coi.is_none());
    }

    #[test]
    fn test_find_closest_coi_index_all_same_distance() {
        // if the distance is the same for all cois, take the first one
        let cois = create_pos_cois(&[[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]]);
        let embedding = arr1(&[1., 1., 1.]).into();
        let (index, _) = CoiSystem::default()
            .find_closest_coi_index(&embedding, &cois)
            .unwrap();
        assert_eq!(index, 0);
    }

    #[test]
    fn test_update_coi_add_point() {
        let mut cois = create_pos_cois(&[[30., 0., 0.], [0., 20., 0.], [0., 0., 40.]]);
        let embedding = arr1(&[1., 1., 1.]).into();

        let config = Configuration::default();
        let threshold = config.threshold;

        let coi_system = CoiSystem::new(config);
        let (index, distance) = coi_system
            .find_closest_coi_index(&embedding, &cois)
            .unwrap();

        assert_eq!(index, 1);
        assert_approx_eq!(f32, distance, 26.747852);
        assert!(threshold < distance);

        cois = coi_system.update_coi(&embedding, cois);
        assert_eq!(cois.len(), 4);
    }

    #[test]
    fn test_update_coi_update_point() {
        let cois = create_pos_cois(&[[1., 1., 1.], [10., 10., 10.], [20., 20., 20.]]);
        let embedding = arr1(&[2., 3., 4.]).into();

        let cois = CoiSystem::default().update_coi(&embedding, cois);

        assert_eq!(cois.len(), 3);
        assert_eq!(cois[0].point, arr1(&[1.1, 1.2, 1.3]));
        assert_eq!(cois[1].point, arr1(&[10., 10., 10.]));
        assert_eq!(cois[2].point, arr1(&[20., 20., 20.]));
    }

    #[test]
    fn test_shift_coi_point() {
        let coi = PositiveCoi::new(mock_coi_id(0), arr1(&[1., 1., 1.]).into());
        let embedding = arr1(&[2., 3., 4.]).into();

        let updated_coi = CoiSystem::default().shift_coi_point(&embedding, &coi.point);

        assert_eq!(updated_coi, arr1(&[1.1, 1.2, 1.3]));
    }

    #[test]
    fn test_update_coi_threshold_exclusive() {
        let cois = create_pos_cois(&[[0., 0., 0.]]);
        let embedding = arr1(&[0., 0., 12.]).into();

        let cois = CoiSystem::default().update_coi(&embedding, cois);

        assert_eq!(cois.len(), 2);
        assert_eq!(cois[0].point, arr1(&[0., 0., 0.]));
        assert_eq!(cois[1].point, arr1(&[0., 0., 12.]));
    }

    #[test]
    fn test_update_cois_update_the_same_point_twice() {
        // checks that an updated coi is used in the next iteration
        let cois = create_pos_cois(&[[0., 0., 0.]]);
        let documents = create_data_with_mab(&[[0., 0., 4.9], [0., 0., 5.]]);
        let documents = to_vec_of_ref_of!(documents, &dyn CoiSystemData);

        let config = Configuration {
            threshold: 5.,
            ..Default::default()
        };

        let cois = CoiSystem::new(config).update_cois(documents.as_slice(), cois);

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

        let coi_comp = CoiSystem::default()
            .compute_coi_for_embedding(&embedding, &user_interests)
            .unwrap();

        assert_eq!(coi_comp.id, mock_coi_id(2));
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

        let coi_system = CoiSystem::default();
        let coi_comp = coi_system
            .compute_coi_for_embedding(&embedding, &user_interests)
            .unwrap();

        assert_eq!(coi_comp.id, mock_coi_id(2));
        assert_approx_eq!(f32, coi_comp.pos_distance, 4.8904557);
        assert_approx_eq!(f32, coi_comp.neg_distance, f32::MAX, ulps = 0);
    }

    #[test]
    fn test_compute_coi() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(&[[1., 4., 4.], [3., 6., 6.]]);

        let documents_coi = CoiSystem::default()
            .compute_coi(documents, &user_interests)
            .unwrap();

        assert_eq!(documents_coi[0].coi.id, mock_coi_id(1));
        assert_approx_eq!(f32, documents_coi[0].coi.pos_distance, 2.8996046);
        assert_approx_eq!(f32, documents_coi[0].coi.neg_distance, 3.7416575);

        assert_eq!(documents_coi[1].coi.id, mock_coi_id(1));
        assert_approx_eq!(f32, documents_coi[1].coi.pos_distance, 5.8501925);
        assert_approx_eq!(f32, documents_coi[1].coi.neg_distance, SQRT_2);
    }

    #[test]
    #[should_panic(expected = "vectors must consist of real values only")]
    fn test_compute_coi_all_nan() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(&[[NAN, NAN, NAN]]);
        let _ = CoiSystem::default().compute_coi(documents, &user_interests);
    }

    #[test]
    #[should_panic(expected = "vectors must consist of real values only")]
    fn test_compute_coi_single_nan() {
        let positive = create_pos_cois(&[[3., 2., 1.], [1., 2., 3.]]);
        let negative = create_neg_cois(&[[4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(&[[1., NAN, 2.]]);
        let _ = CoiSystem::default().compute_coi(documents, &user_interests);
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
        let documents = create_data_with_mab(&[[1., 4., 4.], [3., 6., 6.], [1., 1., 1.]]);
        let documents = to_vec_of_ref_of!(documents, &dyn CoiSystemData);

        let coi_system = CoiSystem::new(Configuration {
            threshold: 5.0,
            ..Default::default()
        });
        let UserInterests { positive, negative } = coi_system
            .update_user_interests(&history, &documents, user_interests)
            .unwrap();

        assert_eq!(positive.len(), 3);

        assert_approx_eq!(f32, positive[0].alpha, 1.);
        assert_approx_eq!(f32, positive[0].beta, 1.);
        assert_eq!(positive[0].point, arr1(&[2.7999997, 1.9, 1.]));

        assert_approx_eq!(f32, positive[1].alpha, 1.21);
        assert_approx_eq!(f32, positive[1].beta, 1.1);
        assert_eq!(positive[1].point, arr1(&[1., 2., 3.]));

        assert_approx_eq!(f32, positive[2].alpha, 1.);
        assert_approx_eq!(f32, positive[2].beta, 1.);
        assert_eq!(positive[2].point, arr1(&[3., 6., 6.]));

        assert_eq!(negative.len(), 1);
        assert_eq!(negative[0].point, arr1(&[3.6999998, 4.9, 5.7999997]));
    }

    #[test]
    fn test_update_user_interests_no_matches() {
        let error = CoiSystem::default()
            .update_user_interests(&Vec::new(), &Vec::new(), UserInterests::new())
            .err()
            .unwrap();
        let error = error.downcast::<CoiSystemError>().unwrap();

        assert!(matches!(error, CoiSystemError::NoMatchingDocuments));
    }
}
