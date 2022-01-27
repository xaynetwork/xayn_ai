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
    /// Not enough positive cois (expected {expected:?}, has {has:?})
    NotEnoughPositiveCois { expected: u32, has: u32 },
    /// Not enough negative cois (expected {expected:?}, has {has:?})
    NotEnoughNegativeCois { expected: u32, has: u32 },
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

fn compute_score_for_embedding(
    embedding: &Embedding,
    user_interests: &UserInterests,
    relevances: &mut RelevanceMap,
    neighbors: usize,
    horizon: Duration,
    now: SystemTime,
) -> Result<f32, Error> {
    let cois = find_closest_cois(embedding, user_interests, neighbors)
        .ok_or(Error::FailedToFindTheClosestCois)?;

    let pos_decay = compute_coi_decay_factor(horizon, now, cois.pos_last_view);
    let neg_decay = compute_coi_decay_factor(horizon, now, cois.neg_last_view);

    relevances.compute_relevances(&user_interests.positive, horizon);
    let pos_coi_relevance = relevances
        .relevance_for_coi(&cois.pos_id).unwrap(/* we calculate relevance for all positive cois one line above */);

    Ok(cois.pos_distance * pos_decay + pos_coi_relevance - cois.neg_distance * neg_decay)

    Ok(score)
}

fn has_enough_cois(
    user_interests: &UserInterests,
    min_positive_cois: u32,
    min_negative_cois: u32,
) -> Result<(), Error> {
    let pos_cois = user_interests.positive.len() as u32;
    if pos_cois < min_positive_cois {
        return Err(Error::NotEnoughPositiveCois {
            expected: min_positive_cois,
            has: pos_cois,
        });
    }

    let neg_cois = user_interests.negative.len() as u32;
    if neg_cois < min_negative_cois {
        return Err(Error::NotEnoughNegativeCois {
            expected: min_negative_cois,
            has: neg_cois,
        });
    }
    Ok(())
}

/// Computes the score for all documents based on the given information.
///
/// <https://xainag.atlassian.net/wiki/spaces/M2D/pages/2240708609/Discovery+engine+workflow#The-weighting-of-the-CoI>
/// outlines parts of the score calculation.
///
/// # Errors
/// Fails if the required number of positive or negative cois is not present or
/// if coi relevance is not present in the relevances map.
pub(super) fn compute_score_for_docs(
    documents: &mut [impl Document],
    user_interests: &UserInterests,
    relevances: &mut RelevanceMap,
    config: &Configuration,
) -> Result<HashMap<DocumentId, f32>, Error> {
    has_enough_cois(
        user_interests,
        config.min_positive_cois(),
        config.min_negative_cois(),
    )?;

    let now = system_time_now();
    documents
        .iter()
        .map(|document| {
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

        assert!(has_enough_cois(&user_interests, 0, 0).is_ok());
        assert!(matches!(
            has_enough_cois(&user_interests, 1, 0).unwrap_err(),
            Error::NotEnoughPositiveCois {
                has: 0,
                expected: 1
            }
        ));
        assert!(matches!(
            has_enough_cois(&user_interests, 0, 1).unwrap_err(),
            Error::NotEnoughNegativeCois {
                has: 0,
                expected: 1
            }
        ));
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

        let score = compute_score_for_embedding(
            &embedding,
            &user_interests,
            &mut relevances,
            1,
            horizon,
            now,
        )
        .unwrap();
        // 1.1185127 * 0.99999934 + 0.73094887 - 0 * 100 = 12
        assert_approx_eq!(f32, score, 1.8494608, epsilon = 1e-5);
    }

    #[test]
    fn test_compute_score_for_embedding_no_cois() {
        let embedding = arr1(&[0., 0., 0.]).into();
        let horizon = Duration::from_secs_f32(SECONDS_PER_DAY);

        let res = compute_score_for_embedding(
            &embedding,
            &UserInterests::default(),
            &mut RelevanceMap::default(),
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
