use std::time::Duration;

use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::{
    coi::{stats::CoiStats, CoiId},
    embedding::utils::{l2_distance, Embedding},
};

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[obake(version("0.3.0"))]
#[derive(Clone, Debug, Derivative, Deserialize, Serialize)]
#[derivative(PartialEq)]
pub(crate) struct PositiveCoi {
    #[obake(cfg(">=0.0"))]
    pub(super) id: CoiId,
    #[obake(cfg(">=0.0"))]
    pub(super) point: Embedding,
    #[obake(cfg(">=0.3"))]
    #[derivative(PartialEq = "ignore")]
    pub(super) stats: CoiStats,

    // removed fields go below this line
    #[obake(cfg(">=0.0, <0.2"))]
    pub(super) alpha: f32,
    #[obake(cfg(">=0.0, <0.2"))]
    pub(super) beta: f32,
}

impl From<PositiveCoi_v0_0_0> for PositiveCoi_v0_1_0 {
    fn from(coi: PositiveCoi_v0_0_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
            alpha: coi.alpha,
            beta: coi.beta,
        }
    }
}

impl From<PositiveCoi_v0_1_0> for PositiveCoi_v0_2_0 {
    fn from(coi: PositiveCoi_v0_1_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
        }
    }
}

impl From<PositiveCoi_v0_2_0> for PositiveCoi {
    fn from(coi: PositiveCoi_v0_2_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
            stats: CoiStats::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct NegativeCoi {
    pub id: CoiId,
    pub point: Embedding,
}

pub(crate) trait CoiPoint {
    fn new(id: CoiId, point: Embedding, viewed: Duration) -> Self;

    fn id(&self) -> CoiId;

    fn set_id(&mut self, id: CoiId);

    fn point(&self) -> &Embedding;

    fn set_point(&mut self, embedding: Embedding);
}

macro_rules! coi_point_default_impls {
    () => {
        fn id(&self) -> CoiId {
            self.id
        }

        fn set_id(&mut self, id: CoiId) {
            self.id = id;
        }

        fn point(&self) -> &Embedding {
            &self.point
        }

        fn set_point(&mut self, embedding: Embedding) {
            self.point = embedding;
        }
    };
}

#[cfg(test)]
impl CoiPoint for PositiveCoi_v0_0_0 {
    fn new(id: CoiId, point: Embedding, _viewed: Duration) -> Self {
        Self {
            id,
            point,
            alpha: 1.,
            beta: 1.,
        }
    }

    coi_point_default_impls! {}
}

#[cfg(test)]
impl CoiPoint for PositiveCoi_v0_1_0 {
    fn new(id: CoiId, point: Embedding, _viewed: Duration) -> Self {
        Self {
            id,
            point,
            alpha: 1.,
            beta: 1.,
        }
    }

    coi_point_default_impls! {}
}

#[cfg(test)]
impl CoiPoint for PositiveCoi_v0_2_0 {
    fn new(id: CoiId, point: Embedding, _viewed: Duration) -> Self {
        Self { id, point }
    }

    coi_point_default_impls! {}
}

impl CoiPoint for PositiveCoi {
    fn new(id: CoiId, point: Embedding, viewed: Duration) -> Self {
        Self {
            id,
            point,
            stats: CoiStats::new(viewed),
        }
    }

    coi_point_default_impls! {}
}

impl CoiPoint for NegativeCoi {
    fn new(id: CoiId, point: Embedding, _viewed: Duration) -> Self {
        Self { id, point }
    }

    coi_point_default_impls! {}
}

// generic types can't be versioned, but aliasing and proper naming in the proc macro call works
#[allow(non_camel_case_types)]
type PositiveCois_v0_0_0 = Vec<PositiveCoi_v0_0_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_1_0 = Vec<PositiveCoi_v0_1_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_2_0 = Vec<PositiveCoi_v0_2_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_3_0 = Vec<PositiveCoi>;

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[obake(version("0.3.0"))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub(crate) struct UserInterests {
    #[obake(inherit)]
    #[obake(cfg(">=0.0"))]
    pub(crate) positive: PositiveCois,
    #[obake(cfg(">=0.0"))]
    pub(crate) negative: Vec<NegativeCoi>,
}

impl From<UserInterests_v0_0_0> for UserInterests_v0_1_0 {
    fn from(ui: UserInterests_v0_0_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<UserInterests_v0_1_0> for UserInterests_v0_2_0 {
    fn from(ui: UserInterests_v0_1_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<UserInterests_v0_2_0> for UserInterests {
    fn from(ui: UserInterests_v0_2_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}

/// Finds the closest centre of interest (CoI) for the given embedding.
///
/// Returns the index of the CoI along with the weighted distance between the given embedding
/// and the k nearest CoIs. If no CoIs were given, `None` will be returned.
pub(super) fn find_closest_coi_index(
    embedding: &Embedding,
    cois: &[impl CoiPoint],
    neighbors: usize,
) -> Option<(usize, f32)> {
    if cois.is_empty() {
        return None;
    }

    let mut distances = cois
        .iter()
        .map(|coi| l2_distance(embedding.view(), coi.point().view()))
        .enumerate()
        .collect::<Vec<_>>();
    distances.sort_by(|(_, this), (_, other)| this.partial_cmp(other).unwrap());
    let index = distances[0].0;

    let total = distances.iter().map(|(_, distance)| *distance).sum::<f32>();
    let distance = if total > 0.0 {
        distances
            .iter()
            .take(neighbors)
            .zip(distances.iter().take(neighbors).rev())
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
pub(super) fn find_closest_coi<'coi, CP>(
    embedding: &Embedding,
    cois: &'coi [CP],
    neighbors: usize,
) -> Option<(&'coi CP, f32)>
where
    CP: CoiPoint,
{
    let (index, distance) = find_closest_coi_index(embedding, cois, neighbors)?;
    Some((&cois[index], distance))
}

/// Finds the closest CoI for the given embedding.
///
/// Returns a mutable reference to the CoI along with the weighted distance between the given
/// embedding and the k nearest CoIs. If no CoIs were given, `None` will be returned.
pub(super) fn find_closest_coi_mut<'coi, CP>(
    embedding: &Embedding,
    cois: &'coi mut [CP],
    neighbors: usize,
) -> Option<(&'coi mut CP, f32)>
where
    CP: CoiPoint,
{
    let (index, distance) = find_closest_coi_index(embedding, cois, neighbors)?;
    Some((&mut cois[index], distance))
}

#[cfg(test)]
mod tests {
    use std::f32::NAN;

    use ndarray::arr1;

    use crate::coi::utils::tests::create_pos_cois;
    use test_utils::assert_approx_eq;

    use super::*;

    #[test]
    fn test_find_closest_coi_index() {
        let cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let embedding = arr1(&[1., 5., 9.]).into();

        let (index, distance) = find_closest_coi_index(&embedding, &cois, 4).unwrap();

        assert_eq!(index, 1);
        assert_approx_eq!(f32, distance, 5.7716017);
    }

    #[test]
    fn test_find_closest_coi_index_equal() {
        let cois = create_pos_cois(&[[1., 2., 3.]]);
        let embedding = arr1(&[1., 2., 3.]).into();

        let (index, distance) = find_closest_coi_index(&embedding, &cois, 4).unwrap();

        assert_eq!(index, 0);
        assert_approx_eq!(f32, distance, 0.0, ulps = 0);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only")]
    fn test_find_closest_coi_index_all_nan() {
        let cois = create_pos_cois(&[[1., 2., 3.]]);
        let embedding = arr1(&[NAN, NAN, NAN]).into();
        find_closest_coi_index(&embedding, &cois, 4);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only")]
    fn test_find_closest_coi_index_single_nan() {
        let cois = create_pos_cois(&[[1., 2., 3.]]);
        let embedding = arr1(&[1., NAN, 2.]).into();
        find_closest_coi_index(&embedding, &cois, 4);
    }

    #[test]
    fn test_find_closest_coi_index_empty() {
        let embedding = arr1(&[1., 2., 3.]).into();
        let coi = find_closest_coi_index(&embedding, &[] as &[PositiveCoi], 4);
        assert!(coi.is_none());
    }

    #[test]
    fn test_find_closest_coi_index_all_same_distance() {
        // if the distance is the same for all cois, take the first one
        let cois = create_pos_cois(&[[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]]);
        let embedding = arr1(&[1., 1., 1.]).into();
        let (index, _) = find_closest_coi_index(&embedding, &cois, 4).unwrap();
        assert_eq!(index, 0);
    }
}
