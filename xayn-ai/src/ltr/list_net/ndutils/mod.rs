pub mod io;
mod softmax;

use itertools::Itertools;
use ndarray::{Array, ArrayView, Axis, Ix, RemoveAxis, ShapeError};
pub use softmax::*;

/// A python like axis index where negative values can be used to index from the end.
///
/// # Panics
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
pub(crate) fn relative_index(idx: isize, len: usize) -> usize {
    debug_assert!(len <= isize::MAX as usize && idx > isize::MIN);
    let len = len as isize;
    debug_assert!(idx < len && -idx <= len);

    if idx < 0 {
        (len + idx) as usize
    } else {
        idx as usize
    }
}

/// Stacks given inputs on `Axis(0)` and if this is below `min_size` the last row is repeated until `min_size` is reached.
///
/// # Panics
///
/// - Panics is the axis is out of bounds.
pub fn stack_and_fill_by_repeat<A, D>(
    axis: Axis,
    arrays: &[ArrayView<A, D>],
    min_size_on_axis: Ix,
) -> Result<Array<A, D>, ShapeError>
where
    A: Copy,
    D: RemoveAxis,
{
    let size_on_axis: Ix = arrays.iter().map(|view| view.shape()[axis.index()]).sum();

    if size_on_axis >= min_size_on_axis {
        ndarray::stack(axis, arrays)
    } else {
        let mut new_arrays = arrays.iter().map(|view| view.view()).collect_vec();
        let last = arrays
            .last()
            .ok_or_else(|| ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape))?
            .view();
        let size_of_last = last.shape()[axis.index()];
        let filler_shape = {
            let mut shape = last.raw_dim();
            shape[axis.index()] = min_size_on_axis - size_on_axis;
            shape
        };
        let last_row_of_last = last.split_at(axis, size_of_last - 1).1;
        let filler = last_row_of_last.broadcast(filler_shape).unwrap();
        new_arrays.push(filler);
        ndarray::stack(axis, &*new_arrays)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr3;

    use super::*;
    mod relative_index_panics {
        use super::super::*;

        #[should_panic]
        #[test]
        fn test_large_len() {
            relative_index(0, isize::MAX as usize + 1);
        }

        #[should_panic]
        #[test]
        fn test_large_index() {
            relative_index(isize::MAX, 1);
        }

        #[should_panic]
        #[test]
        fn test_out_of_bounds1() {
            relative_index(3, 3);
        }

        #[should_panic]
        #[test]
        fn test_out_of_bounds2() {
            relative_index(10, 3);
        }

        #[should_panic]
        #[test]
        fn test_out_of_bounds3() {
            relative_index(-4, 3);
        }
    }

    #[test]
    fn test_pylike_idx_returns_the_right_idx() {
        assert_eq!(relative_index(1, 10), 1);
        assert_eq!(relative_index(0, 10), 0);
        assert_eq!(relative_index(9, 10), 9);
        assert_eq!(relative_index(-10, 10), 0);
        assert_eq!(relative_index(-1, 10), 9);
        assert_eq!(relative_index(-9, 10), 1);
    }

    #[test]
    fn test_stack_and_fill_does_just_stack_if_there_is_no_need_to_fill() {
        let first = arr3(&[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]);
        let second = arr3(&[[[10, 1], [2, 3]], [[10, 5], [6, 7]]]);
        let third = arr3(&[[[20, 30], [2, 3]], [[20, 30], [6, 7]]]);

        assert_eq!(
            stack_and_fill_by_repeat(Axis(0), &[first.view(), second.view(), third.view()], 2)
                .unwrap(),
            arr3(&[
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[10, 1], [2, 3]],
                [[10, 5], [6, 7]],
                [[20, 30], [2, 3]],
                [[20, 30], [6, 7]],
            ])
        );

        assert_eq!(
            stack_and_fill_by_repeat(Axis(0), &[first.view(), second.view(), third.view()], 6)
                .unwrap(),
            arr3(&[
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[10, 1], [2, 3]],
                [[10, 5], [6, 7]],
                [[20, 30], [2, 3]],
                [[20, 30], [6, 7]],
            ])
        );

        assert_eq!(
            stack_and_fill_by_repeat(Axis(1), &[first.view(), second.view(), third.view()], 2)
                .unwrap(),
            arr3(&[
                [[0, 1], [2, 3], [10, 1], [2, 3], [20, 30], [2, 3]],
                [[4, 5], [6, 7], [10, 5], [6, 7], [20, 30], [6, 7]],
            ])
        );
    }

    #[test]
    fn test_stack_and_fill_fills_by_repeating_the_last_on_given_axis() {
        let first = arr3(&[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]);
        let second = arr3(&[[[10, 1], [2, 3]], [[10, 5], [6, 7]]]);

        assert_eq!(
            stack_and_fill_by_repeat(Axis(0), &[first.view(), second.view()], 6).unwrap(),
            arr3(&[
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
                [[10, 1], [2, 3]],
                [[10, 5], [6, 7]],
                [[10, 5], [6, 7]],
                [[10, 5], [6, 7]],
            ])
        );

        assert_eq!(
            stack_and_fill_by_repeat(Axis(1), &[first.view(), second.view()], 5).unwrap(),
            arr3(&[
                [[0, 1], [2, 3], [10, 1], [2, 3], [2, 3]],
                [[4, 5], [6, 7], [10, 5], [6, 7], [6, 7]],
            ])
        );
    }
}
