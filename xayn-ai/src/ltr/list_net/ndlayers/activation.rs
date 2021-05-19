use ndarray::{ArrayBase, Axis, DataMut, DataOwned, Dimension, NdFloat, RemoveAxis};

use super::super::ndutils::{relative_index, softmax};

use super::ActivationFunction;

/// reLu activation function.
pub(crate) struct Relu;

impl<A> ActivationFunction<A> for Relu
where
    A: NdFloat,
{
    fn apply_to<S, D>(&self, mut input: ArrayBase<S, D>) -> ArrayBase<S, D>
    where
        S: DataOwned<Elem = A> + DataMut<Elem = A>,
        D: Dimension + RemoveAxis,
    {
        input.mapv_inplace(|v| A::max(A::zero(), v));
        input
    }
}

/// Softmax activation function.
///
/// # Panics on usage
///
/// - if the relative axis index is out of bounds
///
/// E.g. you can't use a `Softmax` activation function
/// with an relative axis index of 10 on an array which
/// is 2-dimensional (and as such only has support the
/// relative axis indices 0,1,-1,-2).
pub(crate) struct Softmax {
    rel_axis_idx: isize,
}

impl Default for Softmax {
    /// Defaults to a softmax over the last axis.
    fn default() -> Self {
        Softmax::new(-1)
    }
}

impl Softmax {
    /// Creates a new Softmax activation function which if used runs the softmax over given axis.
    ///
    /// The axis is specified as a relative index, i.e. you can use `-1` to always run softmax
    /// over the last axis.
    pub(crate) fn new(rel_axis_idx: isize) -> Softmax {
        Self { rel_axis_idx }
    }
}

impl<A> ActivationFunction<A> for Softmax
where
    A: NdFloat,
{
    /// Applies the activation function to given array.
    ///
    /// # Panics
    ///
    /// - If the relative axis index is out of bounds this will panic.
    fn apply_to<S, D>(&self, input: ArrayBase<S, D>) -> ArrayBase<S, D>
    where
        S: DataOwned<Elem = A> + DataMut<Elem = A>,
        D: Dimension + RemoveAxis,
    {
        let axis = Axis(relative_index(self.rel_axis_idx, input.ndim()));
        softmax(input, axis)
    }
}

/// Linear activation function.
///
/// Like common this is a identity function used
/// in cases where there no activation function is needed.
pub(crate) struct Linear;

impl<A> ActivationFunction<A> for Linear {
    fn apply_to<S, D>(&self, input: ArrayBase<S, D>) -> ArrayBase<S, D>
    where
        S: DataOwned<Elem = A> + DataMut<Elem = A>,
        D: Dimension + RemoveAxis,
    {
        input
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr3, Axis};

    use super::*;

    #[test]
    fn test_relu_activation_function_works() {
        let relu = Relu;
        let array = arr3(&[
            [[-1.0f32, 2.], [3.5, -4.0]],
            [[3.0, 2.4], [-3.0, -1.2]],
            [[-12.0, -2.0], [2.0, 12.0]],
        ]);
        let expected = arr3(&[
            [[0.0f32, 2.], [3.5, 0.0]],
            [[3.0, 2.4], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 12.0]],
        ]);
        let output = relu.apply_to(array);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    fn test_linear_activation_function_works() {
        let relu = Linear;
        let array = arr3(&[
            [[-1.0f32, 2.], [3.5, -4.0]],
            [[3.0, 2.4], [-3.0, -1.2]],
            [[-12.0, -2.0], [2.0, 12.0]],
        ]);
        let expected = array.clone();
        let output = relu.apply_to(array);
        assert_approx_eq!(f32, output, expected);
    }

    #[test]
    fn test_softmax_activation_function_works() {
        use super::super::super::ndutils::softmax;
        let relu = Softmax::new(-2);
        let array = arr3(&[
            [[-1.0f32, 2.], [3.5, -4.0]],
            [[3.0, 2.4], [-3.0, -1.2]],
            [[-12.0, -2.0], [2.0, 12.0]],
        ]);
        let expected = softmax(array.clone(), Axis(1));
        let output = relu.apply_to(array);
        assert_approx_eq!(f32, output, expected);
    }
}
