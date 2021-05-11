pub mod activation;
mod dense;
pub use dense::*;
use ndarray::{ArrayBase, DataMut, DataOwned, Dimension, RemoveAxis};

/// Trait representing a activation functions.
pub(crate) trait ActivationFunction<A> {
    /// Applies the activation function to given array.
    ///
    /// In most cases this will call `input.mapv_inplace` and
    /// apply some function element wise.
    ///
    /// # Panic
    ///
    /// Wrongly configured activation functions might panic when
    /// called with incompatible inputs.
    ///
    /// I.e. if you want to do the softmax over then 10th axis in
    /// a 2 dimensional array.
    fn apply_to<S, D>(&self, input: ArrayBase<S, D>) -> ArrayBase<S, D>
    where
        S: DataOwned<Elem = A> + DataMut<Elem = A>,
        D: Dimension + RemoveAxis;
}
