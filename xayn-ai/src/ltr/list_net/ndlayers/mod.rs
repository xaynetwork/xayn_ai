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
    /// # Panics
    ///
    /// Wrongly configured activation functions might panic when
    /// called with incompatible inputs.
    ///
    /// For example using a `Softmax` activation function which
    /// should create the `Softmax` over the 10th axis can not
    /// work if the input array only has 2 axes.
    ///
    /// Any activation function for which this can happen should
    /// document it on the type level documentation.
    fn apply_to<S, D>(&self, input: ArrayBase<S, D>) -> ArrayBase<S, D>
    where
        S: DataOwned<Elem = A> + DataMut<Elem = A>,
        D: Dimension + RemoveAxis;
}
