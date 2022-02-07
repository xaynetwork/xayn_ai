use std::{cmp::Ordering, time::Duration};

use crate::embedding::utils::COSINE_SIMILARITY_RANGE;
use displaydoc::Display;
use thiserror::Error;

use crate::utils::{nan_safe_f32_cmp_desc, SECONDS_PER_DAY};

/// The configuration of the ranker.
#[derive(Clone, Debug)]
pub struct Config {
    shift_factor: f32,
    threshold: f32,
    horizon: Duration,
    gamma: f32,
    penalty: Vec<f32>,
    min_positive_cois: usize,
    min_negative_cois: usize,
}

/// Potential errors of the ranker configuration.
#[derive(Copy, Clone, Debug, Display, Error)]
pub enum Error {
    /// Invalid coi shift factor, expected value from the unit interval
    ShiftFactor,
    /// Invalid coi threshold, expected non-negative value
    Threshold,
    /// Invalid coi gamma, expected value from the unit interval
    Gamma,
    /// Invalid coi penalty, expected non-empty, finite and sorted values
    Penalty,
    /// Invalid minimum number of positive cois, expected positive value
    MinPositiveCois,
    /// Invalid minimum number of negative cois, expected positive value
    MinNegativeCois,
}

impl Config {
    /// The shift factor by how much a Coi is shifted towards a new point.
    pub fn shift_factor(&self) -> f32 {
        self.shift_factor
    }

    /// Sets the shift factor.
    ///
    /// # Errors
    /// Fails if the shift factor is outside of the unit interval.
    pub fn with_shift_factor(self, shift_factor: f32) -> Result<Self, Error> {
        if (0. ..=1.).contains(&shift_factor) {
            Ok(Self {
                shift_factor,
                ..self
            })
        } else {
            Err(Error::ShiftFactor)
        }
    }

    /// The minimum distance between distinct cois.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Sets the threshold.
    ///
    /// # Errors
    /// Fails if the threshold is not within [`COSINE_SIMILARITY_RANGE`]
    #[allow(dead_code)]
    pub fn with_threshold(self, threshold: f32) -> Result<Self, Error> {
        if COSINE_SIMILARITY_RANGE.contains(&threshold) {
            Ok(Self { threshold, ..self })
        } else {
            Err(Error::Threshold)
        }
    }

    /// The time since the last view after which a coi becomes irrelevant.
    pub fn horizon(&self) -> Duration {
        self.horizon
    }

    /// Sets the horizon.
    pub fn with_horizon(self, horizon: Duration) -> Self {
        Self { horizon, ..self }
    }

    /// The weighting between coi and pairwise candidate similarities in the key phrase selection.
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Sets the gamma.
    ///
    /// # Errors
    /// Fails if the gamma is outside of the unit interval.
    pub fn with_gamma(self, gamma: f32) -> Result<Self, Error> {
        if (0. ..=1.).contains(&gamma) {
            Ok(Self { gamma, ..self })
        } else {
            Err(Error::Gamma)
        }
    }

    /// The penalty for less relevant key phrases of a coi in increasing order (ie. lowest penalty
    /// for the most relevant key phrase first and highest penalty for the least relevant key phrase
    /// last). The length of the penalty also serves as the maximum number of key phrases.
    pub fn penalty(&self) -> &[f32] {
        &self.penalty
    }

    /// Sets the penalty.
    ///
    /// # Errors
    /// Fails if the penalty is empty, has non-finite values or is unsorted.
    pub fn with_penalty(self, penalty: &[f32]) -> Result<Self, Error> {
        // TODO: refactor once slice::is_sorted_by() is stabilized
        fn is_sorted_by(slice: &[f32], compare: impl FnMut(&f32, &f32) -> Ordering) -> bool {
            let mut vector = slice.to_vec();
            vector.sort_unstable_by(compare);
            vector == slice
        }

        if !penalty.is_empty()
            && penalty.iter().copied().all(f32::is_finite)
            && is_sorted_by(penalty, nan_safe_f32_cmp_desc)
        {
            Ok(Self {
                penalty: penalty.to_vec(),
                ..self
            })
        } else {
            Err(Error::Penalty)
        }
    }

    /// The maximum number of key phrases picked during the coi key phrase selection.
    pub fn max_key_phrases(&self) -> usize {
        self.penalty.len()
    }

    /// The minimum number of positive cois required for the context calculation.
    pub(crate) fn min_positive_cois(&self) -> usize {
        self.min_positive_cois
    }

    /// Sets the minimum number of positive cois.
    ///
    /// # Errors
    /// Fails if the minimum number is zero.
    #[allow(dead_code)]
    pub(crate) fn with_min_positive_cois(self, min_positive_cois: usize) -> Result<Self, Error> {
        if min_positive_cois > 0 {
            Ok(Self {
                min_positive_cois,
                ..self
            })
        } else {
            Err(Error::MinPositiveCois)
        }
    }

    /// The minimum number of negative cois required for the context calculation.
    pub(crate) fn min_negative_cois(&self) -> usize {
        self.min_negative_cois
    }

    /// Sets the minimum number of negative cois.
    ///
    /// # Errors
    /// Fails if the minimum number is zero.
    #[allow(dead_code)]
    pub(crate) fn with_min_negative_cois(self, min_negative_cois: usize) -> Result<Self, Error> {
        if min_negative_cois > 0 {
            Ok(Self {
                min_negative_cois,
                ..self
            })
        } else {
            Err(Error::MinNegativeCois)
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            shift_factor: 0.1,
            threshold: 0.67,
            horizon: Duration::from_secs(SECONDS_PER_DAY as u64 * 30),
            gamma: 0.9,
            penalty: vec![1., 0.75, 0.66],
            min_positive_cois: 2,
            min_negative_cois: 2,
        }
    }
}
