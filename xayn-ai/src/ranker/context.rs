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
    ranker::{document::Document, Config},
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
    pos_similarity: f32,
    pos_last_view: SystemTime,

    /// Distance from the closest negative centre of interest
    neg_similarity: f32,
    neg_last_view: SystemTime,
}

fn find_closest_cois(embedding: &Embedding, user_interests: &UserInterests) -> Option<ClosestCois> {
    let (pos_coi, pos_similarity) = find_closest_coi(&user_interests.positive, embedding)?;
    let (neg_coi, neg_similarity) = find_closest_coi(&user_interests.negative, embedding)?;

    ClosestCois {
        pos_id: pos_coi.id(),
        pos_similarity,
        pos_last_view: pos_coi.stats.last_view,
        neg_similarity,
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
    horizon: Duration,
    now: SystemTime,
) -> Result<f32, Error> {
    let cois =
        find_closest_cois(embedding, user_interests).ok_or(Error::FailedToFindTheClosestCois)?;

    let pos_decay = compute_coi_decay_factor(horizon, now, cois.pos_last_view);
    let neg_decay = compute_coi_decay_factor(horizon, now, cois.neg_last_view);

    let pos_coi_relevance = relevances.relevance_for_coi(&cois.pos_id).unwrap();

    let result =
        cois.pos_similarity * pos_decay + pos_coi_relevance - cois.neg_similarity * neg_decay;
    Ok(result.clamp(f32::MIN, f32::MAX)) // Avoid positive or negative infinity
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
    documents: &[impl Document],
    user_interests: &UserInterests,
    relevances: &mut RelevanceMap,
    config: &Config,
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
        let embedding = arr1(&[1., 4., 4.]).into();

        let epoch = SystemTime::UNIX_EPOCH;
        let now = epoch + Duration::from_secs_f32(2. * SECONDS_PER_DAY);

        let mut positive = create_pos_cois(&[[62., 55., 11.], [76., 30., 80.]]);
        positive[0].stats.last_view -= Duration::from_secs_f32(0.5 * SECONDS_PER_DAY);
        positive[1].stats.last_view -= Duration::from_secs_f32(1.5 * SECONDS_PER_DAY);

        let mut negative = create_neg_cois(&[[6., 61., 6.]]);
        negative[0].last_view = epoch;
        let user_interests = UserInterests { positive, negative };

        let mut relevances = RelevanceMap::default();
        let horizon = Duration::from_secs_f32(2. * SECONDS_PER_DAY);

        relevances.compute_relevances(&user_interests.positive, horizon, now);
        let score =
            compute_score_for_embedding(&embedding, &user_interests, &relevances, horizon, now)
                .unwrap();

        let pos_similarity = 0.78551644;
        let pos_decay = 0.99999934;
        let neg_similarity = 0.7744656;
        let neg_decay = 0.;
        let relevance = 0.49999967;
        let expected = pos_similarity * pos_decay + relevance - neg_similarity * neg_decay;
        assert_approx_eq!(f32, score, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_compute_score_for_embedding_no_cois() {
        let embedding = arr1(&[0., 0., 0.]).into();
        let horizon = Duration::from_secs_f32(SECONDS_PER_DAY);

        let res = compute_score_for_embedding(
            &embedding,
            &UserInterests::default(),
            &RelevanceMap::default(),
            horizon,
            system_time_now(),
        );

        assert!(matches!(
            res.unwrap_err(),
            Error::FailedToFindTheClosestCois
        ));
    }
}
