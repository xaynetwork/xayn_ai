pub mod activation;
mod dense;

pub use dense::*;

use ndarray::{ArrayBase, DataMut, DataOwned, Dimension, RemoveAxis};

/// Trait representing an activation function.
pub(crate) trait ActivationFunction<A>: Clone {
    /// Applies the activation function to the given array.
    ///
    /// In most cases this will call `input.mapv_inplace` to
    /// apply some function element wise.
    ///
    /// # Panics
    ///
    /// Wrongly configured activation functions might panic when
    /// called with incompatible inputs.
    ///
    /// For example using a `Softmax` activation function which
    /// should create the `Softmax` over the 10th axis cannot
    /// work if the input array only has 2 axes.
    ///
    /// Any activation function for which this can happen should
    /// document it on the type level documentation.
    fn apply_to<S, D>(&self, input: ArrayBase<S, D>) -> ArrayBase<S, D>
    where
        S: DataOwned<Elem = A> + DataMut<Elem = A>,
        D: Dimension + RemoveAxis;
}
