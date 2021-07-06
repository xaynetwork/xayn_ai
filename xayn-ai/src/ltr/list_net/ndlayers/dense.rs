use std::ops::{AddAssign, DivAssign, MulAssign};

use super::super::ndutils::io::{
    BinParamsWithScope,
    FailedToRetrieveParams,
    UnexpectedNumberOfDimensions,
};
use ndarray::{
    linalg::Dot,
    Array,
    Array1,
    Array2,
    ArrayBase,
    ArrayView,
    Data,
    Dimension,
    Ix1,
    Ix2,
    IxDyn,
    RemoveAxis,
};
use thiserror::Error;

use super::ActivationFunction;

/// Error triggered if two matrices which should be used together are not compatible.
#[derive(Debug, Error)]
#[error("Can't combine {name_left}({shape_left:?}) with {name_right}({shape_right:?}): {hint}")]
pub struct IncompatibleMatrices {
    pub name_left: &'static str,
    pub shape_left: IxDyn,
    pub name_right: &'static str,
    pub shape_right: IxDyn,
    pub hint: &'static str,
}

#[derive(Debug, Error)]
pub enum LoadingDenseFailed {
    #[error(transparent)]
    IncompatibleMatrices(#[from] IncompatibleMatrices),
    #[error(transparent)]
    DimensionMismatch(#[from] UnexpectedNumberOfDimensions),
    #[error(transparent)]
    FailedToRetrieveParams(#[from] FailedToRetrieveParams),
}

/// A dense feed forward network layer.
///
/// This can be used for both 1D and 2D inputs depending
/// on the activation function.
pub(crate) struct Dense<AF>
where
    AF: ActivationFunction<f32>,
{
    weights: Array2<f32>,
    bias: Array1<f32>,
    activation_function: AF,
}

impl<AF> Dense<AF>
where
    AF: ActivationFunction<f32>,
{
    pub(crate) fn load(
        mut params: BinParamsWithScope,
        activation_function: AF,
    ) -> Result<Self, LoadingDenseFailed> {
        let weights = params.take("weights")?;
        let bias = params.take("bias")?;
        Ok(Self::new(weights, bias, activation_function)?)
    }

    pub(crate) fn new(
        weights: Array2<f32>,
        bias: Array1<f32>,
        activation_function: AF,
    ) -> Result<Self, IncompatibleMatrices> {
        if weights.shape()[1] != bias.shape()[0] {
            Err(IncompatibleMatrices {
                name_left: "Dense/weights",
                shape_left: weights.raw_dim().into_dyn(),
                name_right: "Dense/bias",
                shape_right: bias.raw_dim().into_dyn(),
                hint: "expected weights[1] == bias[0] for broadcasting bias add",
            })
        } else {
            Ok(Self {
                weights,
                bias,
                activation_function,
            })
        }
    }

    pub(crate) fn check_in_out_shapes<D>(&self, mut shape: D) -> Result<D, IncompatibleMatrices>
    where
        D: Dimension,
    {
        let ndim = shape.ndim();
        let name_left = "dense/input";
        let name_right = "dense/weights";
        let shape_right = self.weights.raw_dim();
        match ndim {
            1 | 2 => {
                if shape[ndim - 1] != shape_right[0] {
                    Err(IncompatibleMatrices {
                        name_left,
                        shape_left: shape.into_dyn(),
                        name_right,
                        shape_right: shape_right.into_dyn(),
                        hint: "input matrix can't be dot multipled with weight matrix",
                    })
                } else {
                    shape[ndim - 1] = shape_right[1];
                    Ok(shape)
                }
            }
            _ => Err(IncompatibleMatrices {
                name_left,
                shape_left: shape.into_dyn(),
                name_right,
                shape_right: shape_right.into_dyn(),
                hint: "can only use dot product with 1- or 2-dimensional arrays",
            }),
        }
    }

    /// Return the result of applying this dense layer on given inputs.
    ///
    /// If `for_back_propagation` is `true` this will also return the
    /// intermediate result (`z_out`) from before the activation function
    /// was applied.
    ///
    /// If not the `None` is returned instead.
    pub(crate) fn run<S, D>(
        &self,
        input: &ArrayBase<S, D>,
        for_back_propagation: bool,
    ) -> (Array<f32, D>, Option<Array<f32, D>>)
    where
        S: Data<Elem = f32>,
        D: Dimension + RemoveAxis,
        ArrayBase<S, D>: Dot<Array2<f32>, Output = Array<f32, D>>,
    {
        let mut z_out = None;
        let mut out = input.dot(&self.weights);
        out += &self.bias;
        if for_back_propagation {
            z_out = Some(out.clone());
        }
        let y_out = self.activation_function.apply_to(out);
        (y_out, z_out)
    }

    /// This calculates the gradients of a dense layer *without activation function* for a single row of input.
    ///
    /// But as the chained partial gradients are passed in you can just pass in the gradients
    /// from the activation function.
    ///
    /// If multiple row's are provided this need to be run for each row, then
    /// if there are multiple rows because of ...
    ///
    /// 1) batching, you generate the mean of the gradients
    /// 2) shared weights, you sum the gradients (normally, at least in our case)
    ///
    /// Be sure to pass the right gradients down to parent layers, i.e. apply the
    /// sum/mean in gradient decent but pass the i-th gradient set down as previous
    /// gradients for the parent i-th row (as long as you have no merge/split points).
    /// FIXME IxN vs ArrayN
    pub(crate) fn gradients_from_partials_1d(
        &self,
        input: ArrayView<f32, Ix1>,
        partials: ArrayView<f32, Ix1>,
    ) -> DenseGradientSet {
        let weights_nr_rows = self.weights.shape()[0];
        let weights_nr_columns = self.weights.shape()[1];
        assert_eq!(input.shape()[0], weights_nr_rows);
        assert_eq!(partials.shape()[0], weights_nr_columns);

        // The formula for bias at index i is:  `b_i = x_i + b_i` of which the derivative wrt. b_i is `1`
        let bias_gradients = partials.to_owned();

        // For the weight matrix we need the Jacobian Matrix, i.e. the partial derivative of the matrix
        // multiplication wrt. each weight multiplied with the prev gradient associated with it.
        // The derivative of matmul. wrt. `w_{i,j}` is `s_i` and the relevant prev. gradient is
        // `p_j` so the result is:
        //
        // ```
        // J = [
        //  [ p_0 * s_0   p_1 * s_0 … ]
        //  [ p_0 * s_1   p_1 * s_1 … ]
        //  ⋮
        // ]
        // ```
        //
        // With dimensions [i, j].
        let input = input.into_shape((weights_nr_rows, 1)).unwrap();
        let partials = partials.into_shape((1, weights_nr_columns)).unwrap();
        //FIXME source uses `partials @ input` which will result in the same output but
        //      transposed, it also does `W*x` where we do `x*W` but it also seem to be
        //      based on column vectors while we are more based on row vectors. So I guess
        //      we are living in transpose world ;=)
        let weight_gradients = input.dot(&partials);

        DenseGradientSet {
            weight_gradients,
            bias_gradients,
        }
    }

    /// Add given gradients to the weight and bias matrices.
    pub(crate) fn add_gradients(&mut self, gradients: &DenseGradientSet) {
        self.weights += &gradients.weight_gradients;
        self.bias += &gradients.bias_gradients;
    }

    pub(crate) fn weights(&self) -> &Array<f32, Ix2> {
        &self.weights
    }
}

pub(crate) struct DenseGradientSet {
    pub(crate) weight_gradients: Array<f32, Ix2>,
    pub(crate) bias_gradients: Array<f32, Ix1>,
}

impl DenseGradientSet {
    pub(crate) fn no_change_for(dense: &Dense<impl ActivationFunction<f32>>) -> Self {
        DenseGradientSet {
            weight_gradients: Array::zeros(dense.weights.dim()),
            bias_gradients: Array::zeros(dense.bias.dim()),
        }
    }
}

impl AddAssign for DenseGradientSet {
    fn add_assign(&mut self, rhs: Self) {
        self.weight_gradients += &rhs.weight_gradients;
        self.bias_gradients += &rhs.bias_gradients;
    }
}

impl MulAssign<f32> for DenseGradientSet {
    fn mul_assign(&mut self, rhs: f32) {
        self.weight_gradients *= rhs;
        self.bias_gradients *= rhs;
    }
}

impl DivAssign<f32> for DenseGradientSet {
    fn div_assign(&mut self, rhs: f32) {
        self.weight_gradients /= rhs;
        self.bias_gradients /= rhs;
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array1, Array2, IntoDimension};

    use super::super::activation::{Linear, Relu};

    use super::Dense;

    #[test]
    fn test_dense_matrix_for_2d_input() {
        // (features, units) = (3, 2)
        let weights = arr2(&[[1.0f32, 2.], [4., 8.], [3., 0.]]);
        // (units,) = (2,)
        let bias = arr1(&[0.5, 2.0]);
        let dense = Dense::new(weights, bias, Linear).unwrap();

        // (..., features) = (2, 3);
        let inputs = arr2(&[[10., 1., -10.], [0., 10., 0.]]);
        let expected = arr2(&[[-15.5, 30.], [40.5, 82.]]);
        let (res, _) = dense.run(&inputs, false);
        assert_approx_eq!(f32, res, expected);

        // (..., features) = (1, 3);
        let inputs = arr2(&[[0.5, 1.0, -0.5]]);
        let expected = arr2(&[[3.5, 11.]]);
        let (res, _) = dense.run(&inputs, false);
        assert_approx_eq!(f32, res, expected);
    }

    #[test]
    fn test_activation_function_is_called() {
        let weights = arr2(&[[1.0f32, 2.], [4., 8.], [3., 0.]]);
        let bias = arr1(&[0.5, 2.0]);
        let dense = Dense::new(weights, bias, Relu).unwrap();

        let inputs = arr2(&[[10., 1., -10.], [0., 10., 0.]]);
        let expected = arr2(&[[0.0, 30.], [40.5, 82.]]);
        let (res, _) = dense.run(&inputs, false);
        assert_approx_eq!(f32, res, expected);
    }

    #[test]
    fn test_dense_2d_new_panics_if_shapes_do_not_match() {
        let weights = Array2::<f32>::ones((2, 3));
        let bias = Array1::ones((2,));
        assert!(Dense::new(weights, bias, Linear).is_err());
    }

    #[test]
    fn test_dense_matrix_for_1d_input() {
        // (features, units) = (3, 2)
        let weights = arr2(&[[1.0f32, 2.], [4., 8.], [3., 0.]]);
        // (units,) = (2,)
        let bias = arr1(&[0.5, 2.0]);
        let dense = Dense::new(weights, bias, Linear).unwrap();

        // (..., features) = (2, 3);
        let inputs = arr1(&[10., 1., -10.]);
        let expected = arr1(&[-15.5, 30.]);
        let (res, _) = dense.run(&inputs, false);
        assert_approx_eq!(f32, res, expected);

        // (..., features) = (1, 3);
        let inputs = arr1(&[0.5, 1.0, -0.5]);
        let expected = arr1(&[3.5, 11.]);
        let (res, _) = dense.run(&inputs, false);
        assert_approx_eq!(f32, res, expected);
    }

    #[test]
    fn test_dense_1d_new_errors_if_shapes_do_not_match() {
        let weights = Array2::<f32>::ones((2, 3));
        let bias = Array1::ones((2,));
        assert!(Dense::new(weights, bias, Linear).is_err());
    }

    #[test]
    fn test_check_in_out_shape() {
        let weights = Array2::<f32>::ones((5, 4));
        let bias = Array1::<f32>::ones((4,));
        let dense = Dense::new(weights, bias, Linear).unwrap();

        let dim = [10, 5].into_dimension();
        let out_shape = dense.check_in_out_shapes(dim).unwrap();
        assert_eq!(out_shape, [10, 4].into_dimension());

        let dim = [1, 10, 5].into_dimension();
        dense.check_in_out_shapes(dim).unwrap_err();

        let dim = [10, 4].into_dimension();
        dense.check_in_out_shapes(dim).unwrap_err();
    }

    #[test]
    fn returns_z_out_if_needed() {
        let weights = arr2(&[[1.0f32, 2.], [4., 8.], [3., 0.]]);
        let bias = arr1(&[0.5, 2.0]);
        let dense = Dense::new(weights, bias, Relu).unwrap();

        let inputs = arr2(&[[10., 1., -10.], [0., 10., 0.]]);

        let (y_out, z_out) = dense.run(&inputs, true);

        assert_approx_eq!(f32, y_out, arr2(&[[0.0, 30.], [40.5, 82.]]));
        assert_approx_eq!(f32, z_out.unwrap(), arr2(&[[-15.5, 30.], [40.5, 82.]]));
    }
}
