use std::cmp::Ordering;

use super::utils::{
    classify_documents_based_on_user_feedback,
    collect_matching_documents,
    count_coi_ids,
    extend_user_interests_based_on_documents,
    l2_norm,
    update_alpha,
    update_alpha_or_beta,
    update_beta,
    UserInterestsStatus,
};

use crate::{
    data::{
        document_data::{
            CoiComponent,
            DocumentDataWithCoi,
            DocumentDataWithEmbedding,
            DocumentDataWithMab,
        },
        Coi,
        EmbeddingPoint,
        UserInterests,
    },
    reranker_systems,
    DocumentHistory,
    Error,
};

use super::config::Configuration;

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
    fn find_closest_coi_index(
        &self,
        embedding: &EmbeddingPoint,
        cois: &[Coi],
    ) -> Option<(usize, f32)> {
        let index_and_distance = cois
            .iter()
            .enumerate()
            .map(|(i, coi)| (i, l2_norm((&embedding.0 - &coi.point.0).view())))
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
    /// Returns a immutable reference to the CoI along with the distance between
    /// the given embedding and the CoI. If no CoI was found, `None`
    /// will be returned.
    fn find_closest_coi<'coi>(
        &self,
        embedding: &EmbeddingPoint,
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
        embedding: &EmbeddingPoint,
        cois: &'coi mut [Coi],
    ) -> Option<(&'coi mut Coi, f32)> {
        let (index, distance) = self.find_closest_coi_index(embedding, cois)?;
        Some((cois.get_mut(index).unwrap(), distance))
    }

    /// Creates a new CoI that is shifted towards the position of `embedding`.
    fn shift_coi_point(&self, embedding: &EmbeddingPoint, coi: &EmbeddingPoint) -> EmbeddingPoint {
        let updated =
            &coi.0 * (1. - self.config.update_theta) + &embedding.0 * self.config.update_theta;
        EmbeddingPoint(updated)
    }

    /// Updates the CoIs based on the given embedding. If the embedding is closer to the centroid
    /// (lower than [`Configuration.threshold`]), the centroids position gets updated,
    /// otherwise a new centroid is created.
    fn update_coi(&self, embedding: &EmbeddingPoint, mut cois: Vec<Coi>) -> Vec<Coi> {
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

    // Assigns a CoI for the given embedding.
    // Returns `None` if no CoI could be found otherwise it returns the Id of
    // the CoL to along with the positive and negative distance.
    fn compute_coi_for_embedding(
        &self,
        embedding: &EmbeddingPoint,
        user_interests: &UserInterests,
    ) -> Option<CoiComponent> {
        let (coi, pos_distance) = self.find_closest_coi(embedding, &user_interests.positive)?;
        let (_, neg_distance) = self.find_closest_coi(embedding, &user_interests.negative)?;

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
                let center_of_interest = self
                    .compute_coi_for_embedding(&document.embedding.embedding, user_interests)
                    .ok_or(Error {})?;
                Ok(DocumentDataWithCoi::from_document(
                    document,
                    center_of_interest,
                ))
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
            return Err(Error {});
        }

        let (positive_docs, negative_docs) =
            classify_documents_based_on_user_feedback(matching_documents);
        let user_interests =
            extend_user_interests_based_on_documents(positive_docs, negative_docs, user_interests);

        if user_interests.positive.len() < 2 {
            Ok(UserInterestsStatus::NotEnough(user_interests))
        } else {
            Ok(UserInterestsStatus::Ready(user_interests))
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
            return Err(Error {});
        }

        let (positive_docs, negative_docs) =
            classify_documents_based_on_user_feedback(matching_documents);

        user_interests.positive = self.update_cois(&positive_docs, user_interests.positive);
        user_interests.negative = self.update_cois(&negative_docs, user_interests.negative);

        let pos_coi_id_map = count_coi_ids(&positive_docs);

        user_interests.positive =
            update_alpha_or_beta(&pos_coi_id_map, user_interests.positive, update_alpha);
        user_interests.negative =
            update_alpha_or_beta(&pos_coi_id_map, user_interests.negative, update_beta);

        Ok(user_interests)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn find_closest_coi_index() {
        let config = Configuration {
            ..Default::default()
        };

        let coi_1 = Coi::new(0, EmbeddingPoint(array![1., 2., 3.].into_dyn()));
        let coi_2 = Coi::new(1, EmbeddingPoint(array![4., 5., 6.].into_dyn()));
        let coi_3 = Coi::new(2, EmbeddingPoint(array![7., 8., 9.].into_dyn()));

        let cois = vec![coi_1, coi_2, coi_3];

        let point = EmbeddingPoint(array![1., 5., 9.].into_dyn());

        let coi_system = CoiSystem::new(config);

        let (index, _distance) = coi_system.find_closest_coi_index(&point, &cois).unwrap();
        assert_eq!(index, 1);
    }

    #[test]
    fn add_point() {
        let config = Configuration {
            ..Default::default()
        };

        let coi_1 = Coi::new(0, EmbeddingPoint(array![20., 0., 0.].into_dyn()));
        let coi_2 = Coi::new(1, EmbeddingPoint(array![0., 20., 0.].into_dyn()));
        let coi_3 = Coi::new(2, EmbeddingPoint(array![0., 0., 20.].into_dyn()));

        let mut cois = vec![coi_1, coi_2, coi_3];

        let point = EmbeddingPoint(array![1., 1., 1.].into_dyn());

        let coi_system = CoiSystem::new(config);

        let (index, _distance) = coi_system.find_closest_coi_index(&point, &cois).unwrap();
        assert_eq!(index, 0);

        cois = coi_system.update_coi(&point, cois);

        assert_eq!(cois.len(), 4)
    }

    #[test]
    fn update_point() {
        let config = Configuration {
            ..Default::default()
        };

        let coi_1 = Coi::new(0, EmbeddingPoint(array![1., 0., 0.].into_dyn()));
        let coi_2 = Coi::new(1, EmbeddingPoint(array![0., 1., 0.].into_dyn()));
        let coi_3 = Coi::new(2, EmbeddingPoint(array![0., 0., 1.].into_dyn()));

        let cois = vec![coi_1, coi_2, coi_3];

        let point = EmbeddingPoint(array![1., 1., 1.].into_dyn());

        let coi_system = CoiSystem::new(config);

        let (index, _distance) = coi_system.find_closest_coi_index(&point, &cois).unwrap();
        assert_eq!(index, 0);
        assert_eq!(cois.len(), 3);
    }
}
