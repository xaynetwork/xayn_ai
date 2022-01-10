use std::{num::NonZeroUsize, time::Duration};

use crate::utils::SECONDS_PER_DAY;

#[derive(Clone, Copy)]
pub(crate) struct Configuration {
    /// The shift factor by how much a Coi is shifted towards a new point.
    pub shift_factor: f32,
    /// The minimum distance between distinct cois.
    pub threshold: f32,
    /// The positive number of neighbors for the k-nearest-neighbors distance.
    pub neighbors: NonZeroUsize,
    /// The time since the last view after which a coi becomes irrelevant.
    #[allow(dead_code)]
    pub horizon: Duration,
    /// The maximum number of key phrases picked during the coi key phrase selection. A coi may have
    /// more key phrases than this, eg because of merging.
    #[allow(dead_code)]
    pub max_key_phrases: usize,
    /// The weighting between coi and pairwise candidate similarites in the key phrase selection.
    #[allow(dead_code)]
    pub gamma: f32,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            shift_factor: 0.1,
            threshold: 12.0,
            neighbors: NonZeroUsize::new(4).unwrap(),
            horizon: Duration::from_secs(SECONDS_PER_DAY as u64 * 30),
            max_key_phrases: 3,
            gamma: 0.9,
        }
    }
}
