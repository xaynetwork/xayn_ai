pub(crate) struct Configuration {
    /// The shift factor by how much a Coi is shifted towards a new point.
    pub shift_factor: f32,
    pub threshold: f32,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            shift_factor: 0.1,
            threshold: 12.0,
        }
    }
}
