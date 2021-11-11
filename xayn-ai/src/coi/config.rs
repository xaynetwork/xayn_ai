use std::num::NonZeroUsize;

pub(crate) struct Configuration {
    /// The shift factor by how much a Coi is shifted towards a new point.
    pub(crate) shift_factor: f32,
    pub(crate) threshold: f32,
    /// The positive number of neighbors for the k-nearest-neighbors distance.
    pub(crate) neighbors: NonZeroUsize,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            shift_factor: 0.1,
            threshold: 12.0,
            neighbors: NonZeroUsize::new(4).unwrap(),
        }
    }
}
