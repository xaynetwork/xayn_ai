use std::time::SystemTime;

use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::{
    coi::{stats::CoiStats, CoiId},
    embedding::utils::{l2_distance, Embedding},
    utils::{nan_safe_f32_cmp, system_time_now},
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
    pub(crate) stats: CoiStats,

    // removed fields go below this line
    #[obake(cfg(">=0.0, <0.2"))]
    pub(super) alpha: f32,
    #[obake(cfg(">=0.0, <0.2"))]
    pub(super) beta: f32,
}

impl PositiveCoi {
    pub(crate) fn new(id: impl Into<CoiId>, point: impl Into<Embedding>) -> Self {
        Self {
            id: id.into(),
            point: point.into(),
            stats: CoiStats::new(),
        }
    }
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

#[derive(Clone, Debug, Derivative, Deserialize, Serialize)]
#[derivative(PartialEq)]
pub(crate) struct NegativeCoi {
    pub(super) id: CoiId,
    pub(super) point: Embedding,
    #[derivative(PartialEq = "ignore")]
    pub(crate) last_view: SystemTime,
}

impl NegativeCoi {
    pub(crate) fn new(id: impl Into<CoiId>, point: impl Into<Embedding>) -> Self {
        Self {
            id: id.into(),
            point: point.into(),
            last_view: system_time_now(),
        }
    }
}

pub(crate) trait CoiPoint {
    /// Gets the coi id.
    fn id(&self) -> CoiId;

    /// Gets the coi point.
    fn point(&self) -> &Embedding;

    /// Shifts the coi point towards another point by a factor.
    fn shift_point(&mut self, towards: &Embedding, shift_factor: f32);
}

macro_rules! impl_coi_point {
    ($($(#[$attr:meta])* $coi:ty),* $(,)?) => {
        $(
            $(#[$attr])*
            impl CoiPoint for $coi {
                fn id(&self) -> CoiId {
                    self.id
                }

                fn point(&self) -> &Embedding {
                    &self.point
                }

                fn shift_point(&mut self, towards: &Embedding, shift_factor: f32) {
                    self.point *= 1. - shift_factor;
                    self.point += towards * shift_factor;
                }
            }
        )*
    };
}

impl_coi_point! {
    #[cfg(test)] PositiveCoi_v0_0_0,
    #[cfg(test)] PositiveCoi_v0_1_0,
    #[cfg(test)] PositiveCoi_v0_2_0,
    PositiveCoi,
    NegativeCoi,
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
    cois: &[impl CoiPoint],
    embedding: &Embedding,
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
    distances.sort_by(|(_, this), (_, other)| nan_safe_f32_cmp(this, other));
    let index = distances[0].0;

    let total = distances.iter().map(|&(_, distance)| distance).sum::<f32>();
    let distance = (total > 0.)
        .then(|| {
            distances
                .iter()
                .take(neighbors)
                .zip(distances.iter().take(neighbors).rev())
                .map(|((_, distance), (_, reversed))| distance * (reversed / total))
                .sum()
        })
        .unwrap_or_default();

    Some((index, distance))
}

/// Finds the closest CoI for the given embedding.
///
/// Returns an immutable reference to the CoI along with the weighted distance between the given
/// embedding and the k nearest CoIs. If no CoIs were given, `None` will be returned.
pub(crate) fn find_closest_coi<'coi, CP>(
    cois: &'coi [CP],
    embedding: &Embedding,
    neighbors: usize,
) -> Option<(&'coi CP, f32)>
where
    CP: CoiPoint,
{
    find_closest_coi_index(cois, embedding, neighbors)
        .map(|(index, distance)| (&cois[index], distance))
}

/// Finds the closest CoI for the given embedding.
///
/// Returns a mutable reference to the CoI along with the weighted distance between the given
/// embedding and the k nearest CoIs. If no CoIs were given, `None` will be returned.
pub(super) fn find_closest_coi_mut<'coi, CP>(
    cois: &'coi mut [CP],
    embedding: &Embedding,
    neighbors: usize,
) -> Option<(&'coi mut CP, f32)>
where
    CP: CoiPoint,
{
    find_closest_coi_index(cois, embedding, neighbors)
        .map(move |(index, distance)| (&mut cois[index], distance))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{f32::NAN, time::Duration};

    use ndarray::arr1;

    use crate::coi::utils::tests::create_pos_cois;
    use test_utils::assert_approx_eq;

    use super::*;

    pub(crate) trait CoiPointConstructor {
        fn new(id: impl Into<CoiId>, point: impl Into<Embedding>, viewed: Duration) -> Self;
    }

    impl CoiPointConstructor for PositiveCoi_v0_0_0 {
        fn new(id: impl Into<CoiId>, point: impl Into<Embedding>, _viewed: Duration) -> Self {
            Self {
                id: id.into(),
                point: point.into(),
                alpha: 1.,
                beta: 1.,
            }
        }
    }

    impl CoiPointConstructor for PositiveCoi_v0_1_0 {
        fn new(id: impl Into<CoiId>, point: impl Into<Embedding>, _viewed: Duration) -> Self {
            Self {
                id: id.into(),
                point: point.into(),
                alpha: 1.,
                beta: 1.,
            }
        }
    }

    impl CoiPointConstructor for PositiveCoi_v0_2_0 {
        fn new(id: impl Into<CoiId>, point: impl Into<Embedding>, _viewed: Duration) -> Self {
            Self {
                id: id.into(),
                point: point.into(),
            }
        }
    }

    impl CoiPointConstructor for PositiveCoi {
        fn new(id: impl Into<CoiId>, point: impl Into<Embedding>, _viewed: Duration) -> Self {
            Self::new(id, point)
        }
    }

    impl CoiPointConstructor for NegativeCoi {
        fn new(id: impl Into<CoiId>, point: impl Into<Embedding>, _viewed: Duration) -> Self {
            Self::new(id, point)
        }
    }

    #[test]
    fn test_shift_coi_point() {
        let mut cois = create_pos_cois(&[[1., 1., 1.]]);
        let towards = arr1(&[2., 3., 4.]).into();
        let shift_factor = 0.1;

        cois[0].shift_point(&towards, shift_factor);
        assert_eq!(cois[0].point, arr1(&[1.1, 1.2, 1.3]));
    }

    #[test]
    fn test_find_closest_coi_index() {
        let cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let embedding = arr1(&[1., 5., 9.]).into();

        let (index, distance) = find_closest_coi_index(&cois, &embedding, 4).unwrap();

        assert_eq!(index, 1);
        assert_approx_eq!(f32, distance, 5.7716017);
    }

    #[test]
    fn test_find_closest_coi_index_equal() {
        let cois = create_pos_cois(&[[1., 2., 3.]]);
        let embedding = arr1(&[1., 2., 3.]).into();

        let (index, distance) = find_closest_coi_index(&cois, &embedding, 4).unwrap();

        assert_eq!(index, 0);
        assert_approx_eq!(f32, distance, 0.0, ulps = 0);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only")]
    fn test_find_closest_coi_index_all_nan() {
        let cois = create_pos_cois(&[[1., 2., 3.]]);
        let embedding = arr1(&[NAN, NAN, NAN]).into();
        find_closest_coi_index(&cois, &embedding, 4);
    }

    #[test]
    #[should_panic(expected = "vector must consist of real values only")]
    fn test_find_closest_coi_index_single_nan() {
        let cois = create_pos_cois(&[[1., 2., 3.]]);
        let embedding = arr1(&[1., NAN, 2.]).into();
        find_closest_coi_index(&cois, &embedding, 4);
    }

    #[test]
    fn test_find_closest_coi_index_empty() {
        let embedding = arr1(&[1., 2., 3.]).into();
        let coi = find_closest_coi_index(&[] as &[PositiveCoi], &embedding, 4);
        assert!(coi.is_none());
    }

    #[test]
    fn test_find_closest_coi_index_all_same_distance() {
        // if the distance is the same for all cois, take the first one
        let cois = create_pos_cois(&[[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]]);
        let embedding = arr1(&[1., 1., 1.]).into();
        let (index, _) = find_closest_coi_index(&cois, &embedding, 4).unwrap();
        assert_eq!(index, 0);
    }
}
