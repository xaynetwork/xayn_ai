use std::num::NonZeroUsize;

pub(crate) struct Configuration {
    /// The shift factor by how much a Coi is shifted towards a new point.
    pub shift_factor: f32,
    /// The minimum distance between distinct cois.
    pub threshold: f32,
    /// The positive number of neighbors for the k-nearest-neighbors distance.
    pub neighbors: NonZeroUsize,
    /// The maximum number of key phrases associated with a coi.
    pub max_key_phrases: usize,
    /// The weighting between coi and pairwise candidate similarites in the key phrase selection.
    pub gamma: f32,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            shift_factor: 0.1,
            threshold: 12.0,
            neighbors: NonZeroUsize::new(4).unwrap(),
            max_key_phrases: 3,
            gamma: 0.9,
        }
    }
}
