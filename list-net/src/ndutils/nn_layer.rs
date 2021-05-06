use ndarray::{
    linalg::Dot,
    Array,
    Array1,
    Array2,
    ArrayBase,
    Data,
    IntoDimension,
    Ix1,
    Ix2,
    LinalgScalar,
};

pub trait Loader<A>: Sized {
    fn for_scope(&self, scope: &str) -> Self;
    fn load_matrix<D>(&self, name: &str, dim: D) -> Result<Array<A, D::Dim>, LoadingFailed>
    where
        D: IntoDimension;
}

//TODO error
#[derive(Debug)]
pub struct LoadingFailed;

pub trait ActivationFunction<Array> {
    fn apply_to(&self, input: Array) -> Array;
}

impl<T> ActivationFunction<T> for fn(T) -> T {
    fn apply_to(&self, input: T) -> T {
        (self)(input)
    }
}

pub struct Dense2D<A, AF = fn(Array2<A>) -> Array2<A>>
where
    AF: ActivationFunction<Array2<A>>,
    A: LinalgScalar,
{
    weights: Array2<A>,
    bias: Array1<A>,
    activation_function: AF,
}

impl<A, AF> Dense2D<A, AF>
where
    AF: ActivationFunction<Array2<A>>,
    A: LinalgScalar,
{
    ///
    /// # Panic
    ///
    /// Panics if the weight matrix and bias do not match.
    //TODO error
    pub fn new(weights: Array2<A>, bias: Array1<A>, activation_function: AF) -> Self {
        assert_eq!(weights.shape()[1], bias.shape()[0]);
        Self {
            weights,
            bias,
            activation_function,
        }
    }
    // pub const WEIGHT_MATRIX_NAME: &'static str = "weights";
    // pub const BIAS_MATRIX_NAME: &'static str = "bias";

    // pub fn load(prev_units: usize, units: usize,  activation_function: AF, loader: impl Loader<A>) -> Result<Self, LoadingFailed> {
    //     let weights = loader.load_matrix(Self::WEIGHT_MATRIX_NAME, (prev_units, units))?;
    //     let bias = loader.load_matrix(Self::BIAS_MATRIX_NAME, (units,))?;

    //     Ok(Self {
    //         weights,
    //         bias,
    //         activation_function
    //     })
    // }

    // pub fn units(&self) -> usize {
    //     self.weights.shape()[1]
    // }

    pub fn apply_to<S>(&self, array: ArrayBase<S, Ix2>) -> Array2<A>
    where
        ArrayBase<S, Ix2>: Dot<Array2<A>, Output = Array2<A>>,
        S: Data<Elem = A>,
    {
        let h_out = array.dot(&self.weights) + &self.bias;
        self.activation_function.apply_to(h_out)
    }
}

pub struct Dense1D<A, AF = fn(Array1<A>) -> Array1<A>>
where
    AF: ActivationFunction<Array1<A>>,
    A: LinalgScalar,
{
    weights: Array2<A>,
    bias: Array1<A>,
    activation_function: AF,
}

impl<A, AF> Dense1D<A, AF>
where
    AF: ActivationFunction<Array1<A>>,
    A: LinalgScalar,
{
    ///
    /// # Panic
    ///
    /// Panics if the weight matrix and bias do not match.
    pub fn new(weights: Array2<A>, bias: Array1<A>, activation_function: AF) -> Self {
        assert_eq!(weights.shape()[1], bias.shape()[0]);
        Self {
            weights,
            bias,
            activation_function,
        }
    }

    pub fn apply_to<S>(&self, array: ArrayBase<S, Ix1>) -> Array1<A>
    where
        ArrayBase<S, Ix2>: Dot<Array1<A>, Output = Array1<A>>,
        S: Data<Elem = A>,
    {
        let h_out = array.dot(&self.weights) + &self.bias;
        self.activation_function.apply_to(h_out)
    }
}

pub fn identity<I>(array: I) -> I {
    array
}

#[cfg(test)]
mod tests {
    use std::{convert::identity, panic::catch_unwind};

    use ndarray::{arr1, arr2, Array1, Array2};

    use super::{Dense1D, Dense2D};

    #[test]
    fn test_dense_matrix_for_2d_input() {
        // (features, units) = (3, 2)
        let weights = arr2(&[[1.0f32, 2.], [4., 8.], [3., 0.]]);
        // (units,) = (2,)
        let bias = arr1(&[0.5, 2.0]);
        let dense = Dense2D::new(weights, bias, identity::<Array2<_>> as fn(_) -> _);

        // (..., features) = (2, 3);
        let inputs = arr2(&[[10., 1., -10.], [0., 10., 0.]]);
        let expected = arr2(&[[-15.5, 30.], [40.5, 82.]]);
        let res = dense.apply_to(inputs);
        assert_ndarray_eq!(f32, res, expected);

        // (..., features) = (1, 3);
        let inputs = arr2(&[[0.5, 1.0, -0.5]]);
        let expected = arr2(&[[3.5, 11.]]);
        let res = dense.apply_to(inputs);
        assert_ndarray_eq!(f32, res, expected);
    }

    #[test]
    fn test_dense_2d_new_panics_if_shapes_do_not_match() {
        let weights = Array2::<f32>::ones((2, 3));
        let bias = Array1::ones((2,));
        catch_unwind(|| {
            Dense2D::new(weights, bias, identity::<Array2<_>> as fn(_) -> _);
        })
        .unwrap_err();
    }
    #[test]
    fn test_dense_matrix_for_1d_input() {
        // (features, units) = (3, 2)
        let weights = arr2(&[[1.0f32, 2.], [4., 8.], [3., 0.]]);
        // (units,) = (2,)
        let bias = arr1(&[0.5, 2.0]);
        let dense = Dense1D::new(weights, bias, identity::<Array1<_>> as fn(_) -> _);

        // (..., features) = (2, 3);
        let inputs = arr1(&[10., 1., -10.]);
        let expected = arr1(&[-15.5, 30.]);
        let res = dense.apply_to(inputs);
        assert_ndarray_eq!(f32, res, expected);

        // (..., features) = (1, 3);
        let inputs = arr1(&[0.5, 1.0, -0.5]);
        let expected = arr1(&[3.5, 11.]);
        let res = dense.apply_to(inputs);
        assert_ndarray_eq!(f32, res, expected);
    }

    #[test]
    fn test_dense_1d_new_panics_if_shapes_do_not_match() {
        let weights = Array2::<f32>::ones((2, 3));
        let bias = Array1::ones((2,));
        catch_unwind(|| {
            Dense2D::new(weights, bias, identity::<Array2<_>> as fn(_) -> _);
        })
        .unwrap_err();
    }
}
