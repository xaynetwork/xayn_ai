use std::time::Duration;

use crate::{coi::CoiError, utils::SECONDS_PER_DAY};

#[derive(Clone)]
pub(crate) struct Configuration {
    shift_factor: f32,
    threshold: f32,
    neighbors: usize,
    horizon: Duration,
    max_key_phrases: usize,
    gamma: f32,
    penalty: Vec<f32>,
}

impl Configuration {
    /// The shift factor by how much a Coi is shifted towards a new point.
    pub(crate) fn shift_factor(&self) -> f32 {
        self.shift_factor
    }

    /// Sets the shift factor for values in the unit interval.
    #[allow(dead_code)]
    pub(crate) fn with_shift_factor(self, shift_factor: f32) -> Result<Self, CoiError> {
        if (0. ..=1.).contains(&shift_factor) {
            Ok(Self {
                shift_factor,
                ..self
            })
        } else {
            Err(CoiError::InvalidShiftFactor(shift_factor))
        }
    }

    /// The minimum distance between distinct cois.
    pub(crate) fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Sets the threshold for non-negative values.
    #[cfg(test)]
    pub(crate) fn with_threshold(self, threshold: f32) -> Result<Self, CoiError> {
        if threshold >= 0. {
            Ok(Self { threshold, ..self })
        } else {
            Err(CoiError::InvalidThreshold(threshold))
        }
    }

    /// The positive number of neighbors for the k-nearest-neighbors distance.
    pub(crate) fn neighbors(&self) -> usize {
        self.neighbors
    }

    /// Sets the neighbors for positive values.
    #[allow(dead_code)]
    pub(crate) fn with_neighbors(self, neighbors: usize) -> Result<Self, CoiError> {
        if neighbors > 0 {
            Ok(Self { neighbors, ..self })
        } else {
            Err(CoiError::InvalidNeighbors(neighbors))
        }
    }

    /// The time since the last view after which a coi becomes irrelevant.
    #[cfg(test)]
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

    /// Sets the gamma for values in the unit interval.
    #[allow(dead_code)]
    pub(crate) fn with_gamma(self, gamma: f32) -> Result<Self, CoiError> {
        if (0. ..=1.).contains(&gamma) {
            Ok(Self { gamma, ..self })
        } else {
            Err(CoiError::InvalidGamma(gamma))
        }
    }

    /// The penalty for less relevant key phrases of a coi in increasing order (ie. lowest penalty
    /// for the most relevant key phrase first and highest penalty for the least relevant key phrase
    /// last). The length of the penalty also serves as the maximum number of key phrases.
    #[cfg(test)]
    pub(crate) fn penalty(&self) -> &[f32] {
        self.penalty.as_slice()
    }

    /// Sets the penalty for non-empty, finite values.
    #[allow(dead_code)]
    pub(crate) fn with_penalty(self, penalty: &[f32]) -> Result<Self, CoiError> {
        let penalty = penalty.to_vec();
        if !penalty.is_empty() && penalty.iter().copied().all(f32::is_finite) {
            Ok(Self { penalty, ..self })
        } else {
            Err(CoiError::InvalidPenalty(penalty))
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
