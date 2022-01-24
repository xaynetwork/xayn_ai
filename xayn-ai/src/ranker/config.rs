use std::{cmp::Ordering, time::Duration};

use displaydoc::Display;
use thiserror::Error;

use crate::utils::{nan_safe_f32_cmp_desc, SECONDS_PER_DAY};

/// The configuration of the ranker.
#[derive(Clone, Debug)]
pub(crate) struct Configuration {
    shift_factor: f32,
    threshold: f32,
    neighbors: usize,
    horizon: Duration,
    max_key_phrases: usize,
    gamma: f32,
    penalty: Vec<f32>,
}

/// Potential errors of the ranker configuration.
#[derive(Copy, Clone, Debug, Display, Error)]
pub(crate) enum Error {
    /// Invalid coi shift factor, expected value from the unit interval
    ShiftFactor,
    /// Invalid coi threshold, expected non-negative value
    #[cfg(test)]
    Threshold,
    /// Invalid coi neighbors, expected positive value
    Neighbors,
    /// Invalid coi gamma, expected value from the unit interval
    Gamma,
    /// Invalid coi penalty, expected non-empty, finite and sorted values
    Penalty,
}

impl Configuration {
    /// The shift factor by how much a Coi is shifted towards a new point.
    pub(crate) fn shift_factor(&self) -> f32 {
        self.shift_factor
    }

    /// Sets the shift factor.
    ///
    /// # Errors
    /// Fails if the shift factor is outside of the unit interval.
    #[allow(dead_code)]
    pub(crate) fn with_shift_factor(self, shift_factor: f32) -> Result<Self, Error> {
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
    pub(crate) fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Sets the threshold.
    ///
    /// # Errors
    /// Fails if the threshold is negative.
    #[cfg(test)]
    pub(crate) fn with_threshold(self, threshold: f32) -> Result<Self, Error> {
        if threshold >= 0. {
            Ok(Self { threshold, ..self })
        } else {
            Err(Error::Threshold)
        }
    }

    /// The positive number of neighbors for the k-nearest-neighbors distance.
    pub(crate) fn neighbors(&self) -> usize {
        self.neighbors
    }

    /// Sets the neighbors.
    ///
    /// # Errors
    /// Fails if the neighbors is zero.
    #[allow(dead_code)]
    pub(crate) fn with_neighbors(self, neighbors: usize) -> Result<Self, Error> {
        if neighbors > 0 {
            Ok(Self { neighbors, ..self })
        } else {
            Err(Error::Neighbors)
        }
    }

    /// The time since the last view after which a coi becomes irrelevant.
    pub(crate) fn horizon(&self) -> Duration {
        self.horizon
    }

    /// Sets the horizon.
    #[cfg(test)]
    pub(crate) fn with_horizon(self, horizon: Duration) -> Self {
        Self { horizon, ..self }
    }

    /// The weighting between coi and pairwise candidate similarites in the key phrase selection.
    pub(crate) fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Sets the gamma.
    ///
    /// # Errors
    /// Fails if the gamma is outside of the unit interval.
    #[allow(dead_code)]
    pub(crate) fn with_gamma(self, gamma: f32) -> Result<Self, Error> {
        if (0. ..=1.).contains(&gamma) {
            Ok(Self { gamma, ..self })
        } else {
            Err(Error::Gamma)
        }
    }

    /// The penalty for less relevant key phrases of a coi in increasing order (ie. lowest penalty
    /// for the most relevant key phrase first and highest penalty for the least relevant key phrase
    /// last). The length of the penalty also serves as the maximum number of key phrases.
    pub(crate) fn penalty(&self) -> &[f32] {
        &self.penalty
    }

    /// Sets the penalty.
    ///
    /// # Errors
    /// Fails if the penalty is empty, has non-finite values or is unsorted.
    #[allow(dead_code)]
    pub(crate) fn with_penalty(self, penalty: &[f32]) -> Result<Self, Error> {
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
    pub(crate) fn max_key_phrases(&self) -> usize {
        self.penalty.len()
    }
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            shift_factor: 0.1,
            threshold: 12.0,
            neighbors: 4,
            horizon: Duration::from_secs(SECONDS_PER_DAY as u64 * 30),
            max_key_phrases: 3,
            gamma: 0.9,
            penalty: vec![1., 0.75, 0.66],
        }
    }
}
