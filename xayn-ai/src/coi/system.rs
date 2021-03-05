use std::{cmp::Ordering, ops::Deref};

use thiserror::Error;

use super::{
    config::Configuration,
    utils::{
        classify_documents_based_on_user_feedback,
        collect_matching_documents,
        count_coi_ids,
        extend_user_interests_based_on_documents,
        l2_norm,
        update_alpha,
        update_beta,
        UserInterestsStatus,
    },
};
use crate::{
    bert::Embedding,
    data::{
        document_data::{
            CoiComponent,
            DocumentDataWithCoi,
            DocumentDataWithEmbedding,
            DocumentDataWithMab,
        },
        Coi,
        UserInterests,
    },
    reranker_systems,
    DocumentHistory,
    Error,
};

#[derive(Error, Debug)]
pub enum CoiSystemError {
    #[error("No CoI could be found for the given embedding")]
    NoCoi,
    #[error("No matching documents could be found.")]
    NoMatchingDocuments,
}

pub struct CoiSystem {
    config: Configuration,
}

impl Default for CoiSystem {
    fn default() -> Self {
        Self::new(Configuration::default())
    }
}

impl CoiSystem {
    /// Creates a new centre of interest system.
    pub fn new(config: Configuration) -> Self {
        Self { config }
    }

    /// Finds the closest centre of interest (CoI) for the given embedding.
    /// Returns the index of the CoI along with the distance between
    /// the given embedding and the CoI. If no CoI was found, `None`
    /// will be returned.
    fn find_closest_coi_index(&self, embedding: &Embedding, cois: &[Coi]) -> Option<(usize, f32)> {
        let index_and_distance = cois
            .iter()
            .enumerate()
            .map(|(i, coi)| (i, l2_norm(embedding.deref() - coi.point.deref())))
            .fold(
                (None, f32::MAX),
                |acc, (i, b)| match PartialOrd::partial_cmp(&acc.1, &b) {
                    Some(Ordering::Greater) => (Some(i), b),
                    _ => acc,
                },
            );
        match index_and_distance {
            (Some(index), distance) => Some((index, distance)),
            _ => None,
        }
    }

    /// Finds the closest CoI for the given embedding.
    /// Returns an immutable reference to the CoI along with the distance between
    /// the given embedding and the CoI. If no CoI was found, `None`
    /// will be returned.
    fn find_closest_coi<'coi>(
        &self,
        embedding: &Embedding,
        cois: &'coi [Coi],
    ) -> Option<(&'coi Coi, f32)> {
        let (index, distance) = self.find_closest_coi_index(embedding, cois)?;
        Some((cois.get(index).unwrap(), distance))
    }

    /// Finds the closest CoI for the given embedding.
    /// Returns a mutable reference to the CoI along with the distance between
    /// the given embedding and the CoI. If no CoI was found, `None`
    /// will be returned.
    fn find_closest_coi_mut<'coi>(
        &self,
        embedding: &Embedding,
        cois: &'coi mut [Coi],
    ) -> Option<(&'coi mut Coi, f32)> {
        let (index, distance) = self.find_closest_coi_index(embedding, cois)?;
        Some((cois.get_mut(index).unwrap(), distance))
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
    fn update_coi(&self, embedding: &Embedding, mut cois: Vec<Coi>) -> Vec<Coi> {
        match self.find_closest_coi_mut(embedding, &mut cois) {
            Some((coi, distance)) if distance < self.config.threshold => {
                coi.point = self.shift_coi_point(embedding, &coi.point);
            }
            _ => cois.push(Coi::new(cois.len() + 1, embedding.clone())),
        }
        cois
    }

    /// Updates the CoIs based on the embeddings of docs.
    fn update_cois(&self, docs: &[&DocumentDataWithMab], cois: Vec<Coi>) -> Vec<Coi> {
        docs.iter().fold(cois, |cois, doc| {
            self.update_coi(&doc.embedding.embedding, cois)
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

impl reranker_systems::CoiSystem for CoiSystem {
    fn compute_coi(
        &self,
        documents: Vec<DocumentDataWithEmbedding>,
        user_interests: &UserInterests,
    ) -> Result<Vec<DocumentDataWithCoi>, Error> {
        documents
            .into_iter()
            .map(|document| {
                let coi = self
                    .compute_coi_for_embedding(&document.embedding.embedding, user_interests)
                    .ok_or(CoiSystemError::NoCoi)?;
                Ok(DocumentDataWithCoi::from_document(document, coi))
            })
            .collect()
    }

    fn make_user_interests(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithEmbedding],
        user_interests: UserInterests,
    ) -> Result<UserInterestsStatus, Error> {
        let matching_documents = collect_matching_documents(history, documents);

        if matching_documents.is_empty() {
            return Err(CoiSystemError::NoMatchingDocuments.into());
        }

        let (positive_docs, negative_docs) =
            classify_documents_based_on_user_feedback(matching_documents);
        let user_interests =
            extend_user_interests_based_on_documents(positive_docs, negative_docs, user_interests);

        if user_interests.positive.len() >= 2 {
            Ok(UserInterestsStatus::Ready(user_interests))
        } else {
            Ok(UserInterestsStatus::NotEnough(user_interests))
        }
    }

    fn update_user_interests(
        &self,
        history: &[DocumentHistory],
        documents: &[DocumentDataWithMab],
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

        let pos_coi_id_map = count_coi_ids(&positive_docs);

        user_interests.positive = update_alpha(&pos_coi_id_map, user_interests.positive);
        user_interests.negative = update_beta(&pos_coi_id_map, user_interests.negative);

        Ok(user_interests)
    }
}

#[cfg(test)]
mod tests {
    use std::f32::{consts::SQRT_2, NAN};

    use float_cmp::approx_eq;
    use ndarray::{array, Array1};

    use super::*;

    use crate::{
        coi::utils::tests::{
            create_cois,
            create_data_with_embeddings,
            create_document_history,
            create_embedding,
        },
        data::{
            document::{DocumentId, Relevance, UserFeedback},
            document_data::{
                ContextComponent,
                DocumentDataWithMab,
                DocumentIdComponent,
                EmbeddingComponent,
                LtrComponent,
                MabComponent,
            },
            CoiId,
        },
        reranker_systems::CoiSystem as CoiSystemTrait,
    };

    pub fn create_data_with_mab(points: Vec<Array1<f32>>) -> Vec<DocumentDataWithMab> {
        points
            .into_iter()
            .enumerate()
            .map(|(id, point)| DocumentDataWithMab {
                document_id: DocumentIdComponent {
                    id: DocumentId(id.to_string()),
                },
                embedding: EmbeddingComponent {
                    embedding: create_embedding(point),
                },
                coi: CoiComponent {
                    id: CoiId(1),
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
        let cois = create_cois(vec![
            array![1., 2., 3.],
            array![4., 5., 6.],
            array![7., 8., 9.],
        ]);
        let embedding = create_embedding(array![1., 5., 9.]);

        let (index, distance) = CoiSystem::default()
            .find_closest_coi_index(&embedding, &cois)
            .unwrap();

        assert_eq!(index, 1);
        assert!(approx_eq!(f32, distance, 4.2426405));
    }

    #[test]
    fn test_find_closest_coi_index_nan() {
        let cois = create_cois(vec![array![1., 2., 3.]]);
        let embedding_all_nan = create_embedding(array![NAN, NAN, NAN]);

        let coi = CoiSystem::default().find_closest_coi_index(&embedding_all_nan, &cois);
        assert!(coi.is_none());

        let embedding_single_nan = create_embedding(array![1., NAN, 2.]);
        let coi = CoiSystem::default().find_closest_coi_index(&embedding_single_nan, &cois);
        assert!(coi.is_none());
    }

    #[test]
    fn test_find_closest_coi_index_empty() {
        let embedding = create_embedding(array![1., 2., 3.]);

        let coi = CoiSystem::default().find_closest_coi_index(&embedding, &Vec::new());

        assert!(coi.is_none());
    }

    #[test]
    fn test_find_closest_coi_index_all_same_distance() {
        // if the distance is the same for all cois, take the first one
        let cois = create_cois(vec![
            array![10., 0., 0.],
            array![0., 10., 0.],
            array![0., 0., 10.],
        ]);
        let embedding = create_embedding(array![1., 1., 1.]);

        let coi_system = CoiSystem::default();
        let (index, _) = coi_system
            .find_closest_coi_index(&embedding, &cois)
            .unwrap();

        assert_eq!(index, 0);
    }

    #[test]
    fn test_update_coi_add_point() {
        let mut cois = create_cois(vec![
            array![30., 0., 0.],
            array![0., 20., 0.],
            array![0., 0., 40.],
        ]);
        let embedding = create_embedding(array![1., 1., 1.]);

        let config = Configuration::default();
        let threshold = config.threshold;

        let coi_system = CoiSystem::new(config);
        let (index, distance) = coi_system
            .find_closest_coi_index(&embedding, &cois)
            .unwrap();

        assert_eq!(index, 1);
        assert!(approx_eq!(f32, distance, 19.052559));
        assert!(threshold < distance);

        cois = coi_system.update_coi(&embedding, cois);
        assert_eq!(cois.len(), 4)
    }

    #[test]
    fn test_update_coi_update_point() {
        let cois = create_cois(vec![
            array![1., 1., 1.],
            array![10., 10., 10.],
            array![20., 20., 20.],
        ]);
        let embedding = create_embedding(array![2., 3., 4.]);

        let cois = CoiSystem::default().update_coi(&embedding, cois);

        assert_eq!(cois.len(), 3);
        assert_eq!(cois[0].point, create_embedding(array![1.1, 1.2, 1.3]));
        assert_eq!(cois[1].point, create_embedding(array![10., 10., 10.]));
        assert_eq!(cois[2].point, create_embedding(array![20., 20., 20.]))
    }

    #[test]
    fn test_shift_coi_point() {
        let coi = Coi::new(0, create_embedding(array![1., 1., 1.]));
        let embedding = create_embedding(array![2., 3., 4.]);

        let updated_coi = CoiSystem::default().shift_coi_point(&embedding, &coi.point);

        assert_eq!(updated_coi, create_embedding(array![1.1, 1.2, 1.3]));
    }

    #[test]
    fn test_compute_coi_for_embedding() {
        let positive = create_cois(vec![
            array![1., 0., 0.],
            array![0., 1., 0.],
            array![0., 0., 1.],
        ]);
        let negative = create_cois(vec![
            array![10., 0., 0.],
            array![0., 10., 0.],
            array![0., 0., 10.],
        ]);
        let user_interests = UserInterests { positive, negative };
        let embedding = create_embedding(array![2., 3., 4.]);

        let coi_comp = CoiSystem::default()
            .compute_coi_for_embedding(&embedding, &user_interests)
            .unwrap();

        assert_eq!(coi_comp.id, CoiId(2));
        assert!(approx_eq!(f32, coi_comp.pos_distance, 4.690416));
        assert!(approx_eq!(f32, coi_comp.neg_distance, 7.));
    }

    #[test]
    fn test_compute_coi_for_embedding_empty_negative_cois() {
        let positive_cois = create_cois(vec![
            array![1., 0., 0.],
            array![0., 1., 0.],
            array![0., 0., 1.],
        ]);
        let user_interests = UserInterests {
            positive: positive_cois,
            negative: Vec::new(),
        };
        let embedding = create_embedding(array![2., 3., 4.]);

        let coi_system = CoiSystem::default();
        let coi_comp = coi_system
            .compute_coi_for_embedding(&embedding, &user_interests)
            .unwrap();

        assert_eq!(coi_comp.id, CoiId(2));
        assert!(approx_eq!(f32, coi_comp.pos_distance, 4.690416));
        assert!(approx_eq!(f32, coi_comp.neg_distance, f32::MAX));
    }

    #[test]
    fn test_make_user_interests_ready() {
        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Irrelevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);
        let documents = create_data_with_embeddings(vec![
            array![1., 2., 3.],
            array![3., 2., 1.],
            array![4., 5., 6.],
        ]);

        let status = CoiSystem::default()
            .make_user_interests(&history, &documents, UserInterests::new())
            .unwrap();

        let UserInterests { positive, negative } = match status {
            UserInterestsStatus::NotEnough(_) => panic!("status should be Ready"),
            UserInterestsStatus::Ready(interests) => interests,
        };

        assert_eq!(positive[0].id.0, 0);
        assert_eq!(positive[0].point, create_embedding(array![3., 2., 1.]));
        assert!(approx_eq!(f32, positive[0].alpha, 1.));
        assert!(approx_eq!(f32, positive[0].beta, 1.));

        assert_eq!(positive[1].id.0, 1);
        assert_eq!(positive[1].point, create_embedding(array![4., 5., 6.]));
        assert!(approx_eq!(f32, positive[1].alpha, 1.));
        assert!(approx_eq!(f32, positive[1].beta, 1.));

        assert_eq!(negative[0].id.0, 0);
        assert_eq!(negative[0].point, create_embedding(array![1., 2., 3.]));
        assert!(approx_eq!(f32, negative[0].alpha, 1.));
        assert!(approx_eq!(f32, negative[0].beta, 1.));
    }

    #[test]
    fn test_make_user_interest_empty_negative_cois() {
        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);
        let documents = create_data_with_embeddings(vec![array![1., 2., 3.], array![3., 2., 1.]]);

        let status = CoiSystem::default()
            .make_user_interests(&history, &documents, UserInterests::new())
            .unwrap();

        let UserInterests { positive, negative } = match status {
            UserInterestsStatus::NotEnough(_) => panic!("status should be Ready"),
            UserInterestsStatus::Ready(interests) => interests,
        };

        assert_eq!(positive[0].id.0, 0);
        assert_eq!(positive[0].point, create_embedding(array![1., 2., 3.]));
        assert!(approx_eq!(f32, positive[0].alpha, 1.));
        assert!(approx_eq!(f32, positive[0].beta, 1.));

        assert_eq!(positive[1].id.0, 1);
        assert_eq!(positive[1].point, create_embedding(array![3., 2., 1.]));
        assert!(approx_eq!(f32, positive[1].alpha, 1.));
        assert!(approx_eq!(f32, positive[1].beta, 1.));

        assert!(negative.is_empty());
    }

    #[test]
    fn test_make_user_interests_not_enough_coi() {
        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Irrelevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);
        let documents = create_data_with_embeddings(vec![array![1., 2., 3.], array![3., 2., 1.]]);

        let status = CoiSystem::default()
            .make_user_interests(&history, &documents, UserInterests::new())
            .unwrap();

        let UserInterests { positive, negative } = match status {
            UserInterestsStatus::NotEnough(interests) => interests,
            UserInterestsStatus::Ready(_) => panic!("status should be NotEnough"),
        };

        assert_eq!(positive[0].id.0, 0);
        assert_eq!(positive[0].point, create_embedding(array![3., 2., 1.]));
        assert!(approx_eq!(f32, positive[0].alpha, 1.));
        assert!(approx_eq!(f32, positive[0].beta, 1.));

        assert_eq!(negative[0].id.0, 0);
        assert_eq!(negative[0].point, create_embedding(array![1., 2., 3.]));
        assert!(approx_eq!(f32, negative[0].alpha, 1.));
        assert!(approx_eq!(f32, negative[0].beta, 1.));
    }

    #[test]
    fn test_make_user_interests_no_matches() {
        let error = CoiSystem::default()
            .make_user_interests(&Vec::new(), &Vec::new(), UserInterests::new())
            .err()
            .unwrap();
        let error = error.downcast::<CoiSystemError>().unwrap();

        assert!(matches!(error, CoiSystemError::NoMatchingDocuments));
    }

    #[test]
    fn test_compute_coi() {
        let positive = create_cois(vec![array![3., 2., 1.], array![1., 2., 3.]]);
        let negative = create_cois(vec![array![4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(vec![array![1., 4., 4.], array![3., 6., 6.]]);

        let documents_coi = CoiSystem::default()
            .compute_coi(documents, &user_interests)
            .unwrap();

        assert_eq!(documents_coi[0].coi.id.0, 1);
        assert!(approx_eq!(f32, documents_coi[0].coi.pos_distance, 2.236068));
        assert!(approx_eq!(
            f32,
            documents_coi[0].coi.neg_distance,
            3.7416575
        ));

        assert_eq!(documents_coi[1].coi.id.0, 1);
        assert!(approx_eq!(
            f32,
            documents_coi[1].coi.pos_distance,
            5.3851647
        ));
        assert!(approx_eq!(f32, documents_coi[1].coi.neg_distance, SQRT_2));
    }

    #[test]
    fn test_compute_coi_nan() {
        let positive = create_cois(vec![array![3., 2., 1.], array![1., 2., 3.]]);
        let negative = create_cois(vec![array![4., 5., 6.]]);
        let user_interests = UserInterests { positive, negative };
        let documents = create_data_with_embeddings(vec![array![NAN, NAN, NAN]]);

        let error = CoiSystem::default()
            .compute_coi(documents, &user_interests)
            .err()
            .unwrap();
        let error = error.downcast::<CoiSystemError>().unwrap();
        assert!(matches!(error, CoiSystemError::NoCoi));

        let documents = create_data_with_embeddings(vec![array![1., NAN, 2.]]);
        let error = CoiSystem::default()
            .compute_coi(documents, &user_interests)
            .err()
            .unwrap();
        let error = error.downcast::<CoiSystemError>().unwrap();
        assert!(matches!(error, CoiSystemError::NoCoi));
    }

    #[test]
    fn test_update_user_interests() {
        let positive = create_cois(vec![array![3., 2., 1.], array![1., 2., 3.]]);
        let negative = create_cois(vec![array![4., 5., 6.]]);

        let user_interests = UserInterests { positive, negative };

        let history = create_document_history(vec![
            (Relevance::Low, UserFeedback::Irrelevant),
            (Relevance::Low, UserFeedback::Relevant),
            (Relevance::Low, UserFeedback::Relevant),
        ]);
        let documents = create_data_with_mab(vec![
            array![1., 4., 4.],
            array![3., 6., 6.],
            array![1., 1., 1.],
        ]);

        let coi_system = CoiSystem::new(Configuration {
            threshold: 5.0,
            ..Default::default()
        });
        let UserInterests { positive, negative } = coi_system
            .update_user_interests(&history, &documents, user_interests)
            .unwrap();

        assert_eq!(positive.len(), 3);

        assert!(approx_eq!(f32, positive[0].alpha, 1.));
        assert!(approx_eq!(f32, positive[0].beta, 1.));
        assert_eq!(
            positive[0].point,
            create_embedding(array![2.7999997, 1.9, 1.])
        );

        assert!(approx_eq!(f32, positive[1].alpha, 1.21));
        assert!(approx_eq!(f32, positive[1].beta, 1.));
        assert_eq!(positive[1].point, create_embedding(array![1., 2., 3.]));

        assert!(approx_eq!(f32, positive[2].alpha, 1.));
        assert!(approx_eq!(f32, positive[2].beta, 1.));
        assert_eq!(positive[2].point, create_embedding(array![3., 6., 6.]));

        assert_eq!(negative.len(), 1);
        assert!(approx_eq!(f32, negative[0].alpha, 1.));
        assert!(approx_eq!(f32, negative[0].beta, 1.));
        assert_eq!(
            negative[0].point,
            create_embedding(array![3.6999998, 4.9, 5.7999997])
        );
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
