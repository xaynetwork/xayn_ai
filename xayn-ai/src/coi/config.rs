use std::num::NonZeroUsize;

pub(crate) struct Configuration {
    /// The shift factor by how much a Coi is shifted towards a new point.
    pub shift_factor: f32,
    pub threshold: f32,
    /// The number of neighbors for the k-nearest-neighbors distance.
    pub neighbors: NonZeroUsize,
    /// A multiplier for the distances of the k-nearest-neighbors.
    pub distance_scale: f32,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            shift_factor: 0.1,
            threshold: 12.0,
            neighbors: NonZeroUsize::new(4).unwrap(),
            distance_scale: 1.0,
        }
    }
}
