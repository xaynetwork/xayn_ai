use std::{
    collections::HashMap,
    time::{Duration, SystemTime},
};

use displaydoc::Display;
use thiserror::Error;

use crate::{
    coi::{
        compute_coi_decay_factor,
        find_closest_coi,
        point::{CoiPoint, UserInterests},
        RelevanceMap,
    },
    embedding::utils::Embedding,
    ranker::{document::Document, Configuration},
    utils::system_time_now,
    CoiId,
    DocumentId,
};

#[derive(Error, Debug, Display)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum Error {
    /// Not enough cois
    NotEnoughCois,
    /// Failed to find the closest cois
    FailedToFindTheClosestCois,
}

/// Helper struct for [`find_closest_cois`].
struct ClosestCois {
    /// The ID of the closest positive centre of interest
    pos_id: CoiId,
    /// Distance from the closest positive centre of interest
    pos_distance: f32,
    pos_last_view: SystemTime,

    /// Distance from the closest negative centre of interest
    neg_distance: f32,
    neg_last_view: SystemTime,
}

fn find_closest_cois(
    embedding: &Embedding,
    user_interests: &UserInterests,
    neighbors: usize,
) -> Option<ClosestCois> {
    let (pos_coi, pos_distance) = find_closest_coi(&user_interests.positive, embedding, neighbors)?;
    let (neg_coi, neg_distance) = find_closest_coi(&user_interests.negative, embedding, neighbors)?;

    ClosestCois {
        pos_id: pos_coi.id(),
        pos_distance,
        pos_last_view: pos_coi.stats.last_view,
        neg_distance,
        neg_last_view: neg_coi.last_view,
    }
    .into()
}

///# Panics
///
/// Panics if the coi relevance is not present in the [`RelevanceMap`].
fn compute_score_for_embedding(
    embedding: &Embedding,
    user_interests: &UserInterests,
    relevances: &RelevanceMap,
    neighbors: usize,
    horizon: Duration,
    now: SystemTime,
) -> Result<f32, Error> {
    let cois = find_closest_cois(embedding, user_interests, neighbors)
        .ok_or(Error::FailedToFindTheClosestCois)?;

    let pos_decay = compute_coi_decay_factor(horizon, now, cois.pos_last_view);
    let neg_decay = compute_coi_decay_factor(horizon, now, cois.neg_last_view);

    let pos_coi_relevance = relevances.relevance_for_coi(&cois.pos_id).unwrap();

    Ok(
        (cois.pos_distance * pos_decay + pos_coi_relevance - cois.neg_distance * neg_decay)
            .max(f32::MIN)
            .min(f32::MAX),
    )
}

fn has_enough_cois(
    user_interests: &UserInterests,
    min_positive_cois: usize,
    min_negative_cois: usize,
) -> bool {
    user_interests.positive.len() >= min_positive_cois
        && user_interests.negative.len() >= min_negative_cois
}

/// Computes the score for all documents based on the given information.
///
/// <https://xainag.atlassian.net/wiki/spaces/M2D/pages/2240708609/Discovery+engine+workflow#The-weighting-of-the-CoI>
/// outlines parts of the score calculation.
///
/// # Errors
/// Fails if the required number of positive or negative cois is not present.
pub(super) fn compute_score_for_docs(
    documents: &mut [impl Document],
    user_interests: &UserInterests,
    relevances: &mut RelevanceMap,
    config: &Configuration,
) -> Result<HashMap<DocumentId, f32>, Error> {
    if !has_enough_cois(
        user_interests,
        config.min_positive_cois(),
        config.min_negative_cois(),
    ) {
        return Err(Error::NotEnoughCois);
    }

    let now = system_time_now();
    relevances.compute_relevances(&user_interests.positive, config.horizon(), now);
    documents
        .iter()
        .map(|document| {
            // `compute_score_for_embedding` cannot panic because we calculate
            // the relevances for all positive cois before
            let score = compute_score_for_embedding(
                document.smbert_embedding(),
                user_interests,
                relevances,
                config.neighbors(),
                config.horizon(),
                now,
            )?;
            Ok((document.id(), score))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use test_utils::assert_approx_eq;

    use crate::{
        coi::{create_neg_cois, create_pos_cois},
        utils::SECONDS_PER_DAY,
    };

    use super::*;

    #[test]
    fn test_has_enough_cois() {
        let user_interests = UserInterests::default();

        assert!(has_enough_cois(&user_interests, 0, 0));
        assert!(!has_enough_cois(&user_interests, 1, 0));
        assert!(!has_enough_cois(&user_interests, 0, 1));
    }

    #[test]
    fn test_compute_score_for_embedding() {
        let embedding = arr1(&[0., 0., 0.]).into();

        let epoch = SystemTime::UNIX_EPOCH;
        let now = epoch + Duration::from_secs_f32(2. * SECONDS_PER_DAY);

        let mut positive = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.]]);
        positive[0].stats.last_view -= Duration::from_secs_f32(0.5 * SECONDS_PER_DAY);
        positive[1].stats.last_view -= Duration::from_secs_f32(1.5 * SECONDS_PER_DAY);

        let mut negative = create_neg_cois(&[[100., 0., 0.]]);
        negative[0].last_view = epoch;
        let user_interests = UserInterests { positive, negative };

        let mut relevances = RelevanceMap::default();
        let horizon = Duration::from_secs_f32(2. * SECONDS_PER_DAY);

        relevances.compute_relevances(&user_interests.positive, horizon, now);
        let score =
            compute_score_for_embedding(&embedding, &user_interests, &relevances, 1, horizon, now)
                .unwrap();
        // 1.1185127 * 0.99999934 + 0.49999967 - 0 * 100 = 1.6185117
        assert_approx_eq!(f32, score, 1.6185117, epsilon = 1e-6);
    }

    #[test]
    fn test_compute_score_for_embedding_no_cois() {
        let embedding = arr1(&[0., 0., 0.]).into();
        let horizon = Duration::from_secs_f32(SECONDS_PER_DAY);

        let res = compute_score_for_embedding(
            &embedding,
            &UserInterests::default(),
            &RelevanceMap::default(),
            1,
            horizon,
            system_time_now(),
        );

        assert!(matches!(
            res.unwrap_err(),
            Error::FailedToFindTheClosestCois
        ));
    }
}
