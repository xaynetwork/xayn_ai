use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::{
    coi::point::{CoiPoint, NegativeCoi, PositiveCoi},
    utils::{system_time_now, SECONDS_PER_DAY},
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(crate) struct CoiStats {
    pub(crate) view_count: usize,
    pub(crate) view_time: Duration,
    pub(crate) last_view: SystemTime,
}

impl CoiStats {
    pub(crate) fn new(viewed: Duration) -> Self {
        Self {
            view_count: 1,
            view_time: viewed,
            last_view: system_time_now(),
        }
    }

    pub(crate) fn update(&mut self, viewed: Duration) {
        self.view_count += 1;
        self.view_time += viewed;
        self.last_view = system_time_now();
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            view_count: self.view_count + other.view_count,
            view_time: self.view_time + other.view_time,
            last_view: self.last_view.max(other.last_view),
        }
    }
}

impl Default for CoiStats {
    fn default() -> Self {
        Self {
            view_count: 1,
            view_time: Duration::ZERO,
            last_view: SystemTime::UNIX_EPOCH,
        }
    }
}

pub(crate) trait CoiPointStats {
    fn stats(&self) -> CoiStats;

    fn update_stats(&mut self, viewed: Duration);
}

impl CoiPointStats for PositiveCoi {
    fn stats(&self) -> CoiStats {
        self.stats
    }

    fn update_stats(&mut self, viewed: Duration) {
        self.stats.update(viewed);
    }
}

impl CoiPointStats for NegativeCoi {
    fn stats(&self) -> CoiStats {
        CoiStats::default()
    }

    fn update_stats(&mut self, _viewed: Duration) {}
}

/// Computes the relevance/weights of the cois.
///
/// The weights are computed from the view counts and view times of each coi and they are not
/// normalized. The horizon specifies the time since the last view after which a coi becomes
/// irrelevant.
#[allow(dead_code)]
pub(super) fn compute_weights<CP>(cois: &[CP], horizon: Duration) -> Vec<f32>
where
    CP: CoiPoint + CoiPointStats,
{
    let counts = cois.iter().map(|coi| coi.stats().view_count).sum::<usize>() as f32 + f32::EPSILON;
    let times = cois
        .iter()
        .map(|coi| coi.stats().view_time)
        .sum::<Duration>()
        .as_secs_f32()
        + f32::EPSILON;
    let now = system_time_now();
    const DAYS_SCALE: f32 = -0.1;
    let horizon = (horizon.as_secs_f32() * DAYS_SCALE / SECONDS_PER_DAY).exp();

    cois.iter()
        .map(|coi| {
            let CoiStats {
                view_count: count,
                view_time: time,
                last_view: last,
                ..
            } = coi.stats();
            let count = count as f32 / counts;
            let time = time.as_secs_f32() / times;
            let days = (now.duration_since(last).unwrap_or_default().as_secs_f32() * DAYS_SCALE
                / SECONDS_PER_DAY)
                .exp();
            let last = ((horizon - days) / (horizon - 1. - f32::EPSILON)).max(0.);
            (count + time) * last
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::coi::utils::tests::create_pos_cois;
    use test_utils::assert_approx_eq;

    use super::*;

    #[test]
    fn test_compute_weights_empty_cois() {
        let cois = create_pos_cois(&[[]]);
        let horizon = Duration::from_secs_f32(SECONDS_PER_DAY);
        let weights = compute_weights(&cois, horizon);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_compute_weights_zero_horizon() {
        let cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.]]);
        let horizon = Duration::ZERO;
        let weights = compute_weights(&cois, horizon);
        assert_approx_eq!(f32, weights, [0., 0.]);
    }

    #[test]
    fn test_compute_weights_count() {
        let mut cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        cois[1].stats.view_count += 1;
        cois[2].stats.view_count += 2;
        let horizon = Duration::from_secs_f32(SECONDS_PER_DAY);
        let weights = compute_weights(&cois, horizon);
        assert_approx_eq!(f32, weights, [0.5, 0.6666667, 0.8333333], epsilon = 0.00001);
    }

    #[test]
    fn test_compute_weights_time() {
        let mut cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        cois[1].stats.view_time += Duration::from_secs(10);
        cois[2].stats.view_time += Duration::from_secs(20);
        let horizon = Duration::from_secs_f32(SECONDS_PER_DAY);
        let weights = compute_weights(&cois, horizon);
        assert_approx_eq!(f32, weights, [0.5, 0.6666667, 0.8333333], epsilon = 0.00001);
    }

    #[test]
    fn test_compute_weights_last() {
        let mut cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        cois[0].stats.last_view -= Duration::from_secs_f32(0.5 * SECONDS_PER_DAY);
        cois[1].stats.last_view -= Duration::from_secs_f32(1.5 * SECONDS_PER_DAY);
        cois[2].stats.last_view -= Duration::from_secs_f32(2.5 * SECONDS_PER_DAY);
        let horizon = Duration::from_secs_f32(2. * SECONDS_PER_DAY);
        let weights = compute_weights(&cois, horizon);
        assert_approx_eq!(
            f32,
            weights,
            [0.48729968, 0.15438259, 0.],
            epsilon = 0.00001,
        );
    }
}
