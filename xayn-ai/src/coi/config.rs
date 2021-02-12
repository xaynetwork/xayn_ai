pub struct Configuration {
    /// parameter for the update over-relaxation.
    pub update_theta: f32,
    pub threshold: f32,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            update_theta: 0.1,
            threshold: 12.0,
        }
    }
}
