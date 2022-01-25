use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::{
    coi::{
        point::PositiveCoi,
        relevance::{Relevance, Relevances},
    },
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

impl PositiveCoi {
    pub(crate) fn update_stats(&mut self, viewed: Duration) {
        self.stats.update(viewed);
    }
}

impl Relevances {
    /// Computes the relevances of the positive cois.
    ///
    /// The relevance of each coi is computed from its view count and view time relative to the
    /// other cois. It's an unnormalized score from the interval `[0, ∞)`.
    ///
    /// The relevances in the maps are replaced by the coi relevances.
    pub(super) fn compute_relevances(&mut self, cois: &[PositiveCoi], horizon: Duration) {
        let counts =
            cois.iter().map(|coi| coi.stats.view_count).sum::<usize>() as f32 + f32::EPSILON;
        let times = cois
            .iter()
            .map(|coi| coi.stats.view_time)
            .sum::<Duration>()
            .as_secs_f32()
            + f32::EPSILON;
        let now = system_time_now();
        const DAYS_SCALE: f32 = -0.1;
        let horizon = (horizon.as_secs_f32() * DAYS_SCALE / SECONDS_PER_DAY).exp();

        for coi in cois {
            let count = coi.stats.view_count as f32 / counts;
            let time = coi.stats.view_time.as_secs_f32() / times;
            let days = (now
                .duration_since(coi.stats.last_view)
                .unwrap_or_default()
                .as_secs_f32()
                * DAYS_SCALE
                / SECONDS_PER_DAY)
                .exp();
            let last = ((horizon - days) / (horizon - 1. - f32::EPSILON)).max(0.);
            let relevance = Relevance::new(((count + time) * last).max(0.).min(f32::MAX)).unwrap(/* finite by construction */);
            self.replace(coi.id, relevance);
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use crate::{
        coi::{key_phrase::KeyPhrase, utils::tests::create_pos_cois},
        ranker::config::Configuration,
    };
    use test_utils::assert_approx_eq;

    use super::*;

    #[test]
    fn test_compute_relevances_empty_cois() {
        let mut relevances = Relevances::default();
        let cois = create_pos_cois(&[[]]);
        let config = Configuration::default();

        relevances.compute_relevances(&cois, config.horizon());
        assert!(relevances.cois_is_empty());
        assert!(relevances.relevances_is_empty());
    }

    #[test]
    fn test_compute_relevances_zero_horizon() {
        let mut relevances = Relevances::default();
        let cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.]]);
        let config = Configuration::default().with_horizon(Duration::ZERO);

        relevances.compute_relevances(&cois, config.horizon());
        assert_eq!(relevances.cois_len(), cois.len());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.]);
        assert_approx_eq!(f32, relevances[cois[1].id], [0.]);
        assert!(relevances.relevances_is_empty());
    }

    #[test]
    fn test_compute_relevances_count() {
        let mut relevances = Relevances::default();
        let mut cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        cois[1].stats.view_count += 1;
        cois[2].stats.view_count += 2;
        let config =
            Configuration::default().with_horizon(Duration::from_secs_f32(SECONDS_PER_DAY));

        relevances.compute_relevances(&cois, config.horizon());
        assert_eq!(relevances.cois_len(), cois.len());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.5], epsilon = 1e-5);
        assert_approx_eq!(f32, relevances[cois[1].id], [0.6666667], epsilon = 1e-5);
        assert_approx_eq!(f32, relevances[cois[2].id], [0.8333333], epsilon = 1e-5);
        assert!(relevances.relevances_is_empty());
    }

    #[test]
    fn test_compute_relevances_time() {
        let mut relevances = Relevances::default();
        let mut cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        cois[1].stats.view_time += Duration::from_secs(10);
        cois[2].stats.view_time += Duration::from_secs(20);
        let config =
            Configuration::default().with_horizon(Duration::from_secs_f32(SECONDS_PER_DAY));

        relevances.compute_relevances(&cois, config.horizon());
        assert_eq!(relevances.cois_len(), cois.len());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.5], epsilon = 1e-5);
        assert_approx_eq!(f32, relevances[cois[1].id], [0.6666667], epsilon = 1e-5);
        assert_approx_eq!(f32, relevances[cois[2].id], [0.8333333], epsilon = 1e-5);
        assert!(relevances.relevances_is_empty());
    }

    #[test]
    fn test_compute_relevances_last() {
        let mut relevances = Relevances::default();
        let mut cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        cois[0].stats.last_view -= Duration::from_secs_f32(0.5 * SECONDS_PER_DAY);
        cois[1].stats.last_view -= Duration::from_secs_f32(1.5 * SECONDS_PER_DAY);
        cois[2].stats.last_view -= Duration::from_secs_f32(2.5 * SECONDS_PER_DAY);
        let config =
            Configuration::default().with_horizon(Duration::from_secs_f32(2. * SECONDS_PER_DAY));

        relevances.compute_relevances(&cois, config.horizon());
        assert_eq!(relevances.cois_len(), cois.len());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.48729968], epsilon = 1e-5);
        assert_approx_eq!(f32, relevances[cois[1].id], [0.15438259], epsilon = 1e-5);
        assert_approx_eq!(f32, relevances[cois[2].id], [0.]);
        assert!(relevances.relevances_is_empty());
    }

    #[test]
    fn test_compute_relevances_with_key_phrases() {
        let cois = create_pos_cois(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let key_phrases = [
            KeyPhrase::new("many", Array1::default((3,))).unwrap(),
            KeyPhrase::new("key", Array1::default((3,))).unwrap(),
            KeyPhrase::new("phrase", Array1::default((3,))).unwrap(),
            KeyPhrase::new("test", Array1::default((3,))).unwrap(),
            KeyPhrase::new("words", Array1::default((3,))).unwrap(),
        ];
        let mut relevances = Relevances::new(
            [
                cois[0].id, cois[0].id, cois[0].id, cois[1].id, cois[1].id, cois[2].id,
            ],
            [0., 0., 1., 0., 0., 0.],
            key_phrases.to_vec(),
        );
        let config = Configuration::default();

        relevances.compute_relevances(&cois, config.horizon());
        assert_eq!(relevances.cois_len(), 3);
        assert_eq!(relevances[cois[0].id].len(), 1);
        assert_eq!(relevances[cois[1].id].len(), 1);
        assert_eq!(relevances[cois[2].id].len(), 1);
        assert_eq!(relevances.relevances_len(), 2);
        let relevance = relevances[cois[0].id].iter().copied().next().unwrap();
        assert_eq!(relevances[(cois[0].id, relevance)], key_phrases[..3]);
        let relevance = relevances[cois[1].id].iter().copied().next().unwrap();
        assert_eq!(relevances[(cois[1].id, relevance)], key_phrases[3..]);
    }
}
