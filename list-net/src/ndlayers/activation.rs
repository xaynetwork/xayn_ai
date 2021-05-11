use ndarray::{ArrayBase, Axis, DataMut, DataOwned, Dimension, NdFloat, RemoveAxis};

use crate::ndutils::{relative_index, softmax};

use super::ActivationFunction;

/// reLu activation function.
///
/// Currently this can't be parametrized and therefore has
/// only the `Default::default()` constructor.
#[derive(Default)]
pub(crate) struct Relu {
    _priv: (),
}

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
/// # Panics
///
/// Using a `Softmax` with a out-of-bounds axis
/// will panic.
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
    ///
    /// # Panics
    ///
    /// See the documentation on the `ActivationFunction` implementation.
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
    /// # Panic
    ///
    /// If the relative index is out of bound or would cause an overflow this will
    /// panic.
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
/// Currently not configurable and as such equivalent to a identity function
/// (like in `keras`).
///
/// Crate new instances using `Default::default()`.
#[derive(Default)]
pub(crate) struct Linear {
    _priv: (),
}

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

    use crate::ndlayers::{
        activation::{Linear, Softmax},
        ActivationFunction,
    };

    use super::Relu;

    #[test]
    fn test_relu_activation_function_works() {
        let relu = Relu::default();
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
        assert_ndarray_eq!(f32, output, expected);
    }

    #[test]
    fn test_linear_activation_function_works() {
        let relu = Linear::default();
        let array = arr3(&[
            [[-1.0f32, 2.], [3.5, -4.0]],
            [[3.0, 2.4], [-3.0, -1.2]],
            [[-12.0, -2.0], [2.0, 12.0]],
        ]);
        let expected = arr3(&[
            [[-1.0f32, 2.], [3.5, -4.0]],
            [[3.0, 2.4], [-3.0, -1.2]],
            [[-12.0, -2.0], [2.0, 12.0]],
        ]);
        let output = relu.apply_to(array);
        assert_ndarray_eq!(f32, output, expected);
    }

    #[test]
    fn test_softmax_activation_function_works() {
        let relu = Softmax::new(-2);
        let array = arr3(&[
            [[-1.0f32, 2.], [3.5, -4.0]],
            [[3.0, 2.4], [-3.0, -1.2]],
            [[-12.0, -2.0], [2.0, 12.0]],
        ]);
        let expected = crate::ndutils::softmax(array.clone(), Axis(1));
        let output = relu.apply_to(array);
        assert_ndarray_eq!(f32, output, expected);
    }
}
