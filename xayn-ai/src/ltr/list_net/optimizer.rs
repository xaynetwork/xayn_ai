use super::{GradientSet, ListNet};

/// Optimizer which applies gradients in a specific way to the current list net instance.
pub trait Optimizer {
    /// Runs the next optimization step by applying the given gradients on the given ListNet.
    //FIXME[follow up PR] this interface doesn't work for all optimizers
    fn apply_gradients(&mut self, list_net: &mut ListNet, gradient_set: GradientSet);
}

/// Mini-Batch Statistic Gradient Descent
pub struct MiniBatchSdg {
    pub learning_rate: f32,
}

impl Optimizer for MiniBatchSdg {
    fn apply_gradients(&mut self, list_net: &mut ListNet, mut gradient_set: GradientSet) {
        gradient_set *= -self.learning_rate;
        list_net.add_gradients(gradient_set)
    }
}
