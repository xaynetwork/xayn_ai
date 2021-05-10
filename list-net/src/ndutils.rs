//! Inspired by  Xayn network : Searchgarden/test_bert/src/ndlayers/

pub mod io;
pub mod nn_layer;
mod softmax;

use ndarray::{Array, ArrayBase, Axis, Data, LinalgScalar, NdFloat, RemoveAxis};
pub use softmax::*;

/// A python like index where negative values can be used to index from the end.
///
/// Do not use this for performance sensitive code, like iteratively indexing an array.
///
/// # Panic
///
/// It's asserted that all of following hold:
///
/// - idx < len (out of bounds)
/// - -idx <= len (index from back out of bounds)
/// - len <= isize::MAX (necessary for overflow check -isize::MIN > isize::MAX)
/// - idx > isize::MIN  (necessary for overflow check)
///
/// if that is not true this will panic.
#[inline]
pub fn pylike_idx(idx: isize, len: usize) -> usize {
    assert!(len <= isize::MAX as usize && idx > isize::MIN);
    let len = len as isize;
    assert!(idx < len && -idx <= len);

    if idx < 0 {
        (len + idx) as usize
    } else {
        idx as usize
    }
}

/// Create a sum over an axis returning a an array with the same number of dimensions.
///
/// The dimension of the axis is set to 1.
///
/// This means it can be used for broadcasting add/sub/mul/etc.
#[inline]
pub fn sum_axis_same_rank<A, S, D>(array: &ArrayBase<S, D>, axis: Axis) -> Array<A, D>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    D: RemoveAxis,
{
    let mut shape = array.raw_dim();
    shape[axis.index()] = 1;

    array.sum_axis(axis).into_shape(shape).unwrap()
}

/// Finds the maximum over an axis returning an array with the same number of dimensions.
///
/// The dimension of the axis is set to 1.
///
/// This means it can be used for broadcasting add/sub/mul/etc.
#[inline]
pub fn max_axis_same_rank<A, S, D>(array: &ArrayBase<S, D>, axis: Axis) -> Array<A, D>
where
    A: NdFloat,
    S: Data<Elem = A>,
    D: RemoveAxis,
{
    let mut shape = array.raw_dim();
    shape[axis.index()] = 1;
    array
        .fold_axis(axis, A::min_value(), |state, val| A::max(*state, *val))
        .into_shape(shape)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

    use ndarray::arr3;

    use super::*;

    #[test]
    fn test_pylike_idx_panics_on_overflow() {
        catch_unwind(|| pylike_idx(0, usize::MAX)).unwrap_err();
        catch_unwind(|| pylike_idx(isize::MAX, 1)).unwrap_err();
    }

    #[test]
    fn test_pylike_idx_panics_on_out_of_bounds() {
        catch_unwind(|| pylike_idx(3, 3)).unwrap_err();
        catch_unwind(|| pylike_idx(10, 3)).unwrap_err();
        catch_unwind(|| pylike_idx(-4, 3)).unwrap_err();
    }

    #[test]
    fn test_pylike_idx_returns_the_right_idx() {
        assert_eq!(pylike_idx(1, 10), 1);
        assert_eq!(pylike_idx(0, 10), 0);
        assert_eq!(pylike_idx(9, 10), 9);
        assert_eq!(pylike_idx(-10, 10), 0);
        assert_eq!(pylike_idx(-1, 10), 9);
        assert_eq!(pylike_idx(-9, 10), 1);
    }

    #[test]
    fn test_max_axis_same_rank_can_be_used_over_all_axis() {
        let array = arr3(&[
            [[1.0f32, 0.], [2., 4.]],
            [[4., 7.], [3., 3.]],
            [[0., 2.], [6., 2.]],
        ]);

        let res = max_axis_same_rank(&array, Axis(0));
        let expected = arr3(&[[[4.0f32, 7.], [6., 4.]]]);
        assert_ndarray_eq!(f32, res, expected, ulps = 0);

        let res = max_axis_same_rank(&array, Axis(1));
        let expected = arr3(&[[[2.0f32, 4.0]], [[4.0, 7.0]], [[6.0, 2.0]]]);
        assert_ndarray_eq!(f32, res, expected, ulps = 0);

        let res = max_axis_same_rank(&array, Axis(2));
        let expected = arr3(&[[[1.0f32], [4.]], [[7.], [3.]], [[2.], [6.]]]);
        assert_ndarray_eq!(f32, res, expected, ulps = 0);
    }

    #[test]
    fn test_sum_axis_same_rank_can_be_used_over_all_axis() {
        let array = arr3(&[
            [[1.0f32, 0.], [2., 4.]],
            [[4., 7.], [3., 3.]],
            [[0., 2.], [6., 2.]],
        ]);

        let res = sum_axis_same_rank(&array, Axis(0));
        let expected = arr3(&[[[5.0f32, 9.], [11., 9.]]]);
        assert_ndarray_eq!(f32, res, expected, ulps = 0);

        let res = sum_axis_same_rank(&array, Axis(1));
        let expected = arr3(&[[[3.0f32, 4.0]], [[7.0, 10.0]], [[6.0, 4.0]]]);
        assert_ndarray_eq!(f32, res, expected, ulps = 0);

        let res = sum_axis_same_rank(&array, Axis(2));
        let expected = arr3(&[[[1.0f32], [6.]], [[11.], [6.]], [[2.], [8.]]]);
        assert_ndarray_eq!(f32, res, expected, ulps = 0);
    }
}
