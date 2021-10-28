use std::ops::{AddAssign, DivAssign, MulAssign};

use displaydoc::Display;
use ndarray::{
    linalg::Dot,
    Array,
    Array1,
    Array2,
    ArrayBase,
    ArrayView1,
    ArrayView2,
    Data,
    Dimension,
    IntoDimension,
    IxDyn,
    RemoveAxis,
};
use thiserror::Error;

use crate::{
    activation::ActivationFunction,
    io::{BinParamsWithScope, FailedToRetrieveParams, UnexpectedNumberOfDimensions},
};

/// Can't combine {name_left}({shape_left:?}) with {name_right}({shape_right:?}): {hint}
#[derive(Debug, Display, Error)]
pub struct IncompatibleMatrices {
    name_left: &'static str,
    shape_left: IxDyn,
    name_right: &'static str,
    shape_right: IxDyn,
    hint: &'static str,
}

impl IncompatibleMatrices {
    pub fn new(
        name_left: &'static str,
        shape_left: impl IntoDimension,
        name_right: &'static str,
        shape_right: impl IntoDimension,
        hint: &'static str,
    ) -> Self {
        Self {
            name_left,
            shape_left: shape_left.into_dimension().into_dyn(),
            name_right,
            shape_right: shape_right.into_dimension().into_dyn(),
            hint,
        }
    }
}

/// Failed to load the Dense layer
#[derive(Debug, Display, Error)]
#[prefix_enum_doc_attributes]
pub enum LoadingDenseFailed {
    /// {0}
    IncompatibleMatrices(#[from] IncompatibleMatrices),
    /// {0}
    DimensionMismatch(#[from] UnexpectedNumberOfDimensions),
    /// {0}
    FailedToRetrieveParams(#[from] FailedToRetrieveParams),
}

/// A dense feed forward network layer.
///
/// This can be used for both 1D and 2D inputs depending
/// on the activation function.
#[derive(Clone)]
pub struct Dense<AF>
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
    pub fn new(
        weights: Array2<f32>,
        bias: Array1<f32>,
        activation_function: AF,
    ) -> Result<Self, IncompatibleMatrices> {
        if weights.shape()[1] == bias.shape()[0] {
            Ok(Self {
                weights,
                bias,
                activation_function,
            })
        } else {
            Err(IncompatibleMatrices {
                name_left: "Dense/weights",
                shape_left: weights.raw_dim().into_dyn(),
                name_right: "Dense/bias",
                shape_right: bias.raw_dim().into_dyn(),
                hint: "expected weights[1] == bias[0] for broadcasting bias add",
            })
        }
    }

    pub fn load(
        mut params: BinParamsWithScope,
        activation_function: AF,
    ) -> Result<Self, LoadingDenseFailed> {
        Self::new(
            params.take("weights")?,
            params.take("bias")?,
            activation_function,
        )
        .map_err(Into::into)
    }

    pub fn weights(&self) -> ArrayView2<f32> {
        self.weights.view()
    }

    pub fn bias(&self) -> ArrayView1<f32> {
        self.bias.view()
    }

    pub fn store_params(self, mut params: BinParamsWithScope) {
        params.insert("weights", self.weights);
        params.insert("bias", self.bias);
    }

    pub fn check_in_out_shapes<D>(&self, mut shape: D) -> Result<D, IncompatibleMatrices>
    where
        D: Dimension,
    {
        let ndim = shape.ndim();
        let name_left = "dense/input";
        let name_right = "dense/weights";
        let shape_right = self.weights.raw_dim();
        if let 1 | 2 = ndim {
            if shape[ndim - 1] == shape_right[0] {
                shape[ndim - 1] = shape_right[1];
                Ok(shape)
            } else {
                Err(IncompatibleMatrices::new(
                    name_left,
                    shape,
                    name_right,
                    shape_right,
                    "input matrix can't be dot multipled with weight matrix",
                ))
            }
        } else {
            Err(IncompatibleMatrices::new(
                name_left,
                shape,
                name_right,
                shape_right,
                "can only use dot product with 1- or 2-dimensional arrays",
            ))
        }
    }

    /// Applies the dense layer on the given inputs.
    ///
    /// If `for_back_propagation` is `true` this will also return the
    /// intermediate result (`z_out`) from before the activation function
    /// was applied. If not then `None` is returned instead.
    pub fn run<S, D>(
        &self,
        input: ArrayBase<S, D>,
        for_back_propagation: bool,
    ) -> (Array<f32, D>, Option<Array<f32, D>>)
    where
        S: Data<Elem = f32>,
        D: Dimension + RemoveAxis,
        ArrayBase<S, D>: Dot<Array2<f32>, Output = Array<f32, D>>,
    {
        let mut out = input.dot(&self.weights);
        out += &self.bias;
        let z_out = for_back_propagation.then(|| out.to_owned());
        let y_out = self.activation_function.apply_to(out);

        (y_out, z_out)
    }

    /// Calculates the gradients of a dense layer based on the relevant partial derivatives.
    ///
    /// # Panics
    ///
    /// The `input` and `partials` arrays are expected to be c- or f-contiguous.
    pub fn gradients_from_partials_1d(
        &self,
        input: ArrayView1<f32>,
        partials: ArrayView1<f32>,
    ) -> DenseGradientSet {
        let weights_nr_rows = self.weights.shape()[0];
        let weights_nr_columns = self.weights.shape()[1];
        assert_eq!(input.shape()[0], weights_nr_rows);
        assert_eq!(partials.shape()[0], weights_nr_columns);

        // The formula for bias at index i is:  `b_i = x_i + b_i` of which the derivative wrt. b_i is `1`
        let bias_gradients = partials.to_owned();

        // For the weight matrix we need the Jacobian Matrix, i.e. denominating the inputs as `s_i`
        // and the partial derivatives as `p_j` we need:
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
        let weight_gradients = input.dot(&partials);

        DenseGradientSet {
            weight_gradients,
            bias_gradients,
        }
    }

    /// Adds given gradients to the weight and bias matrices.
    pub fn add_gradients(&mut self, gradients: &DenseGradientSet) {
        self.weights += &gradients.weight_gradients;
        self.bias += &gradients.bias_gradients;
    }

    /// Divides all parameters (weights, bias) of this dense layer in place.
    pub fn div_parameters_by(&mut self, denominator: f32) {
        self.weights /= denominator;
        self.bias /= denominator;
    }

    /// Adds all parameters of `other` to `self`.
    pub fn add_parameters_of(&mut self, other: Self) {
        let Dense {
            weights,
            bias,
            activation_function: _,
        } = other;
        self.weights += &weights;
        self.bias += &bias;
    }
}

/// A gradient set containing gradients for all parameters in a dense layer.
///
/// (Assuming the activation function has no parameters. It might still have
/// hyper-parameters.)
#[derive(Debug, Clone)]
pub struct DenseGradientSet {
    weight_gradients: Array2<f32>,
    bias_gradients: Array1<f32>,
}

impl DenseGradientSet {
    pub fn new(weight_gradients: Array2<f32>, bias_gradients: Array1<f32>) -> Self {
        Self {
            weight_gradients,
            bias_gradients,
        }
    }

    pub fn weight_gradients(&self) -> &Array2<f32> {
        &self.weight_gradients
    }

    pub fn bias_gradients(&self) -> &Array1<f32> {
        &self.bias_gradients
    }

    /// Merge multiple gradients for the same shared weights.
    ///
    /// This will just sum them up.
    pub fn merge_shared(
        gradients_for_shared_weights: impl IntoIterator<Item = Self>,
    ) -> Option<Self> {
        gradients_for_shared_weights.into_iter().reduce(|mut l, r| {
            l += r;
            l
        })
    }
}

impl AddAssign for DenseGradientSet {
    fn add_assign(&mut self, rhs: Self) {
        debug_assert_eq!(
            (self.weight_gradients.shape(), self.bias_gradients.shape()),
            (rhs.weight_gradients.shape(), rhs.bias_gradients.shape())
        );
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

    use crate::activation::{Linear, Relu};
    use test_utils::assert_approx_eq;

    use super::*;

    impl<'a> AddAssign<&'a Self> for DenseGradientSet {
        fn add_assign(&mut self, rhs: &Self) {
            debug_assert_eq!(
                (self.weight_gradients.shape(), self.bias_gradients.shape()),
                (rhs.weight_gradients.shape(), rhs.bias_gradients.shape())
            );
            self.weight_gradients += &rhs.weight_gradients;
            self.bias_gradients += &rhs.bias_gradients;
        }
    }

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
        let (res, _) = dense.run(inputs, false);
        assert_approx_eq!(f32, res, expected);

        // (..., features) = (1, 3);
        let inputs = arr2(&[[0.5, 1.0, -0.5]]);
        let expected = arr2(&[[3.5, 11.]]);
        let (res, _) = dense.run(inputs, false);
        assert_approx_eq!(f32, res, expected);
    }

    #[test]
    fn test_activation_function_is_called() {
        let weights = arr2(&[[1.0f32, 2.], [4., 8.], [3., 0.]]);
        let bias = arr1(&[0.5, 2.0]);
        let dense = Dense::new(weights, bias, Relu).unwrap();

        let inputs = arr2(&[[10., 1., -10.], [0., 10., 0.]]);
        let expected = arr2(&[[0.0, 30.], [40.5, 82.]]);
        let (res, _) = dense.run(inputs, false);
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
        let (res, _) = dense.run(inputs, false);
        assert_approx_eq!(f32, res, expected);

        // (..., features) = (1, 3);
        let inputs = arr1(&[0.5, 1.0, -0.5]);
        let expected = arr1(&[3.5, 11.]);
        let (res, _) = dense.run(inputs, false);
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

        let (y_out, z_out) = dense.run(inputs, true);

        assert_approx_eq!(f32, y_out, arr2(&[[0.0, 30.], [40.5, 82.]]));
        assert_approx_eq!(f32, z_out.unwrap(), arr2(&[[-15.5, 30.], [40.5, 82.]]));
    }
}
