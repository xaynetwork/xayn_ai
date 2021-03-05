use std::cmp::Ordering;

use rubert::Embeddings;
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
    fn find_closest_coi_index(&self, embedding: &Embeddings, cois: &[Coi]) -> Option<(usize, f32)> {
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
    /// Returns an immutable reference to the CoI along with the distance between
    /// the given embedding and the CoI. If no CoI was found, `None`
    /// will be returned.
    fn find_closest_coi<'coi>(
        &self,
        embedding: &Embeddings,
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
        embedding: &Embeddings,
        cois: &'coi mut [Coi],
    ) -> Option<(&'coi mut Coi, f32)> {
        let (index, distance) = self.find_closest_coi_index(embedding, cois)?;
        Some((cois.get_mut(index).unwrap(), distance))
    }

    /// Creates a new CoI that is shifted towards the position of `embedding`.
    fn shift_coi_point(&self, embedding: &Embeddings, coi: &Embeddings) -> Embeddings {
        let updated =
            &coi.0 * (1. - self.config.shift_factor) + &embedding.0 * self.config.shift_factor;
        Embeddings(updated.into_shared())
    }

    /// Updates the CoIs based on the given embedding. If the embedding is close to the nearest centroid
    /// (within [`Configuration.threshold`]), the centroid's position gets updated,
    /// otherwise a new centroid is created.
    fn update_coi(&self, embedding: &Embeddings, mut cois: Vec<Coi>) -> Vec<Coi> {
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
        embedding: &Embeddings,
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
