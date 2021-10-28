use super::{data::GradientSet, model::ListNet};

/// Optimizer which applies gradients in a specific way to the current list net instance.
pub trait Optimizer {
    /// Runs the next optimization step by applying the given gradients on the given ListNet.
    fn apply_gradients(&mut self, list_net: &mut ListNet, batch_of_gradient_sets: Vec<GradientSet>);
}

/// Mini-Batch Stochastic Gradient Descent
#[derive(Clone)]
pub struct MiniBatchSgd {
    pub learning_rate: f32,
}

impl Optimizer for MiniBatchSgd {
    fn apply_gradients(
        &mut self,
        list_net: &mut ListNet,
        batch_of_gradient_sets: Vec<GradientSet>,
    ) {
        if let Some(mut gradient_set) = GradientSet::mean_of(batch_of_gradient_sets) {
            gradient_set *= -self.learning_rate;
            list_net.add_gradients(gradient_set);
        }
    }
}
