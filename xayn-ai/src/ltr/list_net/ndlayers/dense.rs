use super::super::ndutils::io::{
    BinParamsWithScope,
    FailedToRetrieveParams,
    UnexpectedNumberOfDimensions,
};
use ndarray::{linalg::Dot, Array, Array1, Array2, ArrayBase, Data, Dimension, IxDyn, RemoveAxis};
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

    pub(crate) fn run<S, D>(&self, input: ArrayBase<S, D>) -> Array<f32, D>
    where
        S: Data<Elem = f32>,
        D: Dimension + RemoveAxis,
        ArrayBase<S, D>: Dot<Array2<f32>, Output = Array<f32, D>>,
    {
        let h_out = input.dot(&self.weights) + &self.bias;
        self.activation_function.apply_to(h_out)
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
        let dense = Dense::new(weights, bias, Linear::default()).unwrap();

        // (..., features) = (2, 3);
        let inputs = arr2(&[[10., 1., -10.], [0., 10., 0.]]);
        let expected = arr2(&[[-15.5, 30.], [40.5, 82.]]);
        let res = dense.run(inputs);
        assert_approx_eq!(f32, res, expected);

        // (..., features) = (1, 3);
        let inputs = arr2(&[[0.5, 1.0, -0.5]]);
        let expected = arr2(&[[3.5, 11.]]);
        let res = dense.run(inputs);
        assert_approx_eq!(f32, res, expected);
    }

    #[test]
    fn test_activation_function_is_called() {
        let weights = arr2(&[[1.0f32, 2.], [4., 8.], [3., 0.]]);
        let bias = arr1(&[0.5, 2.0]);
        let dense = Dense::new(weights, bias, Relu::default()).unwrap();

        let inputs = arr2(&[[10., 1., -10.], [0., 10., 0.]]);
        let expected = arr2(&[[0.0, 30.], [40.5, 82.]]);
        let res = dense.run(inputs);
        assert_approx_eq!(f32, res, expected);
    }

    #[test]
    fn test_dense_2d_new_panics_if_shapes_do_not_match() {
        let weights = Array2::<f32>::ones((2, 3));
        let bias = Array1::ones((2,));
        assert!(Dense::new(weights, bias, Linear::default()).is_err());
    }

    #[test]
    fn test_dense_matrix_for_1d_input() {
        // (features, units) = (3, 2)
        let weights = arr2(&[[1.0f32, 2.], [4., 8.], [3., 0.]]);
        // (units,) = (2,)
        let bias = arr1(&[0.5, 2.0]);
        let dense = Dense::new(weights, bias, Linear::default()).unwrap();

        // (..., features) = (2, 3);
        let inputs = arr1(&[10., 1., -10.]);
        let expected = arr1(&[-15.5, 30.]);
        let res = dense.run(inputs);
        assert_approx_eq!(f32, res, expected);

        // (..., features) = (1, 3);
        let inputs = arr1(&[0.5, 1.0, -0.5]);
        let expected = arr1(&[3.5, 11.]);
        let res = dense.run(inputs);
        assert_approx_eq!(f32, res, expected);
    }

    #[test]
    fn test_dense_1d_new_errors_if_shapes_do_not_match() {
        let weights = Array2::<f32>::ones((2, 3));
        let bias = Array1::ones((2,));
        assert!(Dense::new(weights, bias, Linear::default()).is_err());
    }

    #[test]
    fn test_check_in_out_shape() {
        let weights = Array2::<f32>::ones((5, 4));
        let bias = Array1::<f32>::ones((4,));
        let dense = Dense::new(weights, bias, Linear::default()).unwrap();

        let dim = [10, 5].into_dimension();
        let out_shape = dense.check_in_out_shapes(dim).unwrap();
        assert_eq!(out_shape, [10, 4].into_dimension());

        let dim = [1, 10, 5].into_dimension();
        dense.check_in_out_shapes(dim).unwrap_err();

        let dim = [10, 4].into_dimension();
        dense.check_in_out_shapes(dim).unwrap_err();
    }
}
