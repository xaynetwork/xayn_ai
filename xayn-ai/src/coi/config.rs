use std::{cmp::Ordering, time::Duration};

use displaydoc::Display;
use thiserror::Error;

use crate::{
    embedding::utils::COSINE_SIMILARITY_RANGE,
    utils::{nan_safe_f32_cmp_desc, SECONDS_PER_DAY},
};

/// The configuration of the cois.
#[derive(Clone, Debug)]
struct CoiConfig {
    shift_factor: f32,
    threshold: f32,
    min_positive_cois: usize,
    min_negative_cois: usize,
}

impl Default for CoiConfig {
    fn default() -> Self {
        Self {
            shift_factor: 0.1,
            threshold: 0.67,
            min_positive_cois: 2,
            min_negative_cois: 2,
        }
    }
}

/// The configuration of the kpe.
#[derive(Clone, Debug)]
struct KPEConfig {
    horizon: Duration,
    gamma: f32,
    penalty: Vec<f32>,
}

impl Default for KPEConfig {
    fn default() -> Self {
        Self {
            horizon: Duration::from_secs(SECONDS_PER_DAY as u64 * 30),
            gamma: 0.9,
            penalty: vec![1., 0.75, 0.66],
        }
    }
}

/// The configuration of the coi system.
#[derive(Clone, Debug, Default)]
pub struct Config {
    coi: CoiConfig,
    kpe: KPEConfig,
}

/// Errors of the coi system configuration.
#[derive(Copy, Clone, Debug, Display, Error)]
pub enum Error {
    /// Invalid coi shift factor, expected value from the unit interval
    ShiftFactor,
    /// Invalid coi threshold, expected non-negative value
    Threshold,
    /// Invalid minimum number of positive cois, expected positive value
    MinPositiveCois,
    /// Invalid minimum number of negative cois, expected positive value
    MinNegativeCois,
    /// Invalid coi gamma, expected value from the unit interval
    Gamma,
    /// Invalid coi penalty, expected non-empty, finite and sorted values
    Penalty,
}

impl Config {
    /// The shift factor by how much a coi is shifted towards a new point.
    pub fn shift_factor(&self) -> f32 {
        self.coi.shift_factor
    }

    /// Sets the shift factor.
    ///
    /// # Errors
    /// Fails if the shift factor is outside of the unit interval.
    pub fn with_shift_factor(mut self, shift_factor: f32) -> Result<Self, Error> {
        if (0. ..=1.).contains(&shift_factor) {
            self.coi.shift_factor = shift_factor;
            Ok(self)
        } else {
            Err(Error::ShiftFactor)
        }
    }

    /// The minimum distance between distinct cois.
    pub fn threshold(&self) -> f32 {
        self.coi.threshold
    }

    /// Sets the threshold.
    ///
    /// # Errors
    /// Fails if the threshold is not within [`COSINE_SIMILARITY_RANGE`].
    pub fn with_threshold(mut self, threshold: f32) -> Result<Self, Error> {
        if COSINE_SIMILARITY_RANGE.contains(&threshold) {
            self.coi.threshold = threshold;
            Ok(self)
        } else {
            Err(Error::Threshold)
        }
    }

    /// The minimum number of positive cois required for the context calculation.
    pub fn min_positive_cois(&self) -> usize {
        self.coi.min_positive_cois
    }

    /// Sets the minimum number of positive cois.
    ///
    /// # Errors
    /// Fails if the minimum number is zero.
    pub fn with_min_positive_cois(mut self, min_positive_cois: usize) -> Result<Self, Error> {
        if min_positive_cois > 0 {
            self.coi.min_positive_cois = min_positive_cois;
            Ok(self)
        } else {
            Err(Error::MinPositiveCois)
        }
    }

    /// The minimum number of negative cois required for the context calculation.
    pub fn min_negative_cois(&self) -> usize {
        self.coi.min_negative_cois
    }

    /// Sets the minimum number of negative cois.
    ///
    /// # Errors
    /// Fails if the minimum number is zero.
    pub fn with_min_negative_cois(mut self, min_negative_cois: usize) -> Result<Self, Error> {
        if min_negative_cois > 0 {
            self.coi.min_negative_cois = min_negative_cois;
            Ok(self)
        } else {
            Err(Error::MinNegativeCois)
        }
    }

    /// The time since the last view after which a coi becomes irrelevant.
    pub fn horizon(&self) -> Duration {
        self.kpe.horizon
    }

    /// Sets the horizon.
    pub fn with_horizon(mut self, horizon: Duration) -> Self {
        self.kpe.horizon = horizon;
        self
    }

    /// The weighting between coi and pairwise candidate similarities in the key phrase selection.
    pub fn gamma(&self) -> f32 {
        self.kpe.gamma
    }

    /// Sets the gamma.
    ///
    /// # Errors
    /// Fails if the gamma is outside of the unit interval.
    pub fn with_gamma(mut self, gamma: f32) -> Result<Self, Error> {
        if (0. ..=1.).contains(&gamma) {
            self.kpe.gamma = gamma;
            Ok(self)
        } else {
            Err(Error::Gamma)
        }
    }

    /// The penalty for less relevant key phrases of a coi in increasing order (ie. lowest penalty
    /// for the most relevant key phrase first and highest penalty for the least relevant key phrase
    /// last). The length of the penalty also serves as the maximum number of key phrases.
    pub fn penalty(&self) -> &[f32] {
        &self.kpe.penalty
    }

    /// Sets the penalty.
    ///
    /// # Errors
    /// Fails if the penalty is empty, has non-finite values or is unsorted.
    pub fn with_penalty(mut self, penalty: &[f32]) -> Result<Self, Error> {
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
            self.kpe.penalty = penalty.to_vec();
            Ok(self)
        } else {
            Err(Error::Penalty)
        }
    }

    /// The maximum number of key phrases picked during the coi key phrase selection.
    pub fn max_key_phrases(&self) -> usize {
        self.kpe.penalty.len()
    }
}
