use std::cmp::Ordering;

use ndarray::{ArrayBase, Data, Dimension, IntoDimension, Ix};

#[macro_export]
macro_rules! to_vec_of_ref_of {
    ($data: expr, $type:ty) => {
        $data
            .iter()
            .map(|data| -> $type { data })
            .collect::<Vec<_>>()
    };
}

/// Allows comparing and sorting f32 even if `NaN` is involved.
///
/// Pretend that f32 has a total ordering.
///
/// `NaN` is treated as the lowest possible value, similar to what [`f32::max`] does.
///
/// If this is used for sorting this will lead to an ascending order, like
/// for example `[NaN, 0.5, 1.5, 2.0]`.
///
/// By switching the input parameters around this can be used to create a
/// descending sorted order, like e.g.: `[2.0, 1.5, 0.5, NaN]`.
pub(crate) fn nan_safe_f32_cmp(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        // if `partial_cmp` returns None we have at least one `NaN`,
        // we treat it as the lowest value
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, _) => Ordering::Less,
            (_, true) => Ordering::Greater,
            _ => unreachable!("partial_cmp returned None but both numbers are not NaN"),
        }
    })
}

/// `nan_safe_f32_cmp_desc(a,b)` is syntax suggar for `nan_safe_f32_cmp(b, a)`
#[inline]
pub(crate) fn nan_safe_f32_cmp_desc(a: &f32, b: &f32) -> Ordering {
    nan_safe_f32_cmp(b, a)
}

/// Compares two "things" with approximate equality.
///
/// # Examples
///
/// This can be used to compare two floating point numbers:
///
/// ```
/// use xayn_ai::assert_approx_eq;
/// assert_approx_eq!(f32, 0.15039155, 0.1503916, ulps = 3);
/// ```
///
/// Or containers of such:
///
/// ```
/// use xayn_ai::assert_approx_eq;
/// assert_approx_eq!(f32, &[[1.0, 2.], [3., 4.]], vec![[1.0, 2.], [3., 4.]])
/// ```
///
/// Or ndarray arrays:
///
/// ```
/// use ndarray::arr2;
/// use xayn_ai::assert_approx_eq;
/// assert_approx_eq!(
///     f32,
///     arr2(&[[1.0, 2.], [3., 4.]]),
///     arr2(&[[1.0, 2.], [3., 4.]])
/// );
/// ```
///
/// The number of `ulps` defaults to `2` if not specified.
///
/// # Missing Implementations
///
/// Implementations for other primitives, smart pointer types or other sequential containers
/// can easily be added on demand.
///
/// Non sequential containers are not supported.
///
/// # De-Facto `#[cfg(test)]`
///
/// As this is also used by some FFI binding crates, we need to export it and we can't limit it to `#[cfg(test)]`.
///
/// But you can't use this outside of dev/test builds, it won't compile as float-cmp is a dev-only dependency.
/// Furthermore in dependencies which use this you need to have the `float-cmp` dependency available.
#[macro_export]
macro_rules! assert_approx_eq {
    ($t:ty, $left:expr, $right:expr) => {
        assert_approx_eq!($t, $left, $right, ulps = 2)
    };
    ($t:ty, $left:expr, $right:expr, ulps = $ulps:expr) => {{
        let ulps = $ulps;
        let left = $left;
        let right = $right;
        let mut left_iter =
            $crate::ApproxAssertIterHelper::indexed_iter_logical_order(&left, Vec::new());
        let mut right_iter =
            $crate::ApproxAssertIterHelper::indexed_iter_logical_order(&right, Vec::new());
        loop {
            match (left_iter.next(), right_iter.next()) {
                (Some((lidx, lv)), Some((ridx, rv))) => {
                    assert_eq!(
                        lidx, ridx,
                        "Dimensionality mismatch when iterating in logical order: {:?} != {:?}",
                        lidx, ridx
                    );
                    assert!(
                        ::float_cmp::approx_eq!(f32, lv, rv, ulps = ulps),
                        "approximated equal assertion failed (ulps={ulps:?}) at index {idx:?}: {lv:?} == {rv:?}",
                        ulps=ulps,
                        lv=lv,
                        rv=rv,
                        idx=lidx,
                    );
                }
                (Some(pair), None) => {
                    panic!("Left input is longer starting with from index {:?}", pair);
                }
                (None, Some(pair)) => {
                    panic!("Left input is longer starting with from index {:?}", pair);
                }
                (None, None) => break,
            }
        }
    }};
}

/// Helper trait for the `approx_assert_eq!` macro.
///
/// Until we have GAT in rust this is meant to be implemented
/// on a `&`-reference to the thing you want to implement it for.
///
/// This can be implemented for both containers and leaf values (e.g. &f32).
///
/// This trait is tuned for testing, and uses trait objects to reduce the
/// amount of code overhead.
///
/// Only use it for `assert_approx_eq!`.
///
/// # De-Facto `#[cfg(test)]`
///
/// We can't make this `#[cfg(test)]` as we use it in FFI crates,
/// but this should only be used if `#[cfg(test)]` is enabled in
/// either this crate or the dependent of this crate in which
/// you use it.
pub trait ApproxAssertIterHelper<'a>: Copy {
    /// The leaf element, e.g. f32.
    type LeafElement;

    /// Flattened iterates over all leaf elements in this instance.
    ///
    /// The passed in `index_prefix` is the "index" at which
    /// this instance is placed.
    ///
    /// Leaf values implementing this should just return a iterator
    /// which yields a single tuple of their value and the
    /// passed in index prefix.
    ///
    /// Sequential containers are supposed to yield a tuple for each
    /// element in them in which the index is created by pushing
    /// the elements index in this container onto the `index_prefix`.
    fn indexed_iter_logical_order(
        self,
        index_prefix: Vec<Ix>,
    ) -> Box<dyn Iterator<Item = (Vec<Ix>, Self::LeafElement)> + 'a>;
}

impl<'a> ApproxAssertIterHelper<'a> for &'a f32 {
    type LeafElement = f32;

    fn indexed_iter_logical_order(
        self,
        prefix: Vec<Ix>,
    ) -> Box<dyn Iterator<Item = (Vec<Ix>, Self::LeafElement)> + 'a> {
        let iter = std::iter::once((prefix, *self));
        Box::new(iter)
    }
}

impl<'a, T> ApproxAssertIterHelper<'a> for &'a &'a T
where
    &'a T: ApproxAssertIterHelper<'a>,
    T: 'a + ?Sized,
{
    type LeafElement = <&'a T as ApproxAssertIterHelper<'a>>::LeafElement;

    fn indexed_iter_logical_order(
        self,
        prefix: Vec<Ix>,
    ) -> Box<dyn Iterator<Item = (Vec<Ix>, Self::LeafElement)> + 'a> {
        (*self).indexed_iter_logical_order(prefix)
    }
}

impl<'a, T: 'a> ApproxAssertIterHelper<'a> for &'a Vec<T>
where
    &'a T: ApproxAssertIterHelper<'a>,
{
    type LeafElement = <&'a T as ApproxAssertIterHelper<'a>>::LeafElement;

    fn indexed_iter_logical_order(
        self,
        prefix: Vec<Ix>,
    ) -> Box<dyn Iterator<Item = (Vec<Ix>, Self::LeafElement)> + 'a> {
        self.as_slice().indexed_iter_logical_order(prefix)
    }
}

impl<'a, T, const N: usize> ApproxAssertIterHelper<'a> for &'a [T; N]
where
    &'a T: ApproxAssertIterHelper<'a>,
{
    type LeafElement = <&'a T as ApproxAssertIterHelper<'a>>::LeafElement;

    fn indexed_iter_logical_order(
        self,
        prefix: Vec<Ix>,
    ) -> Box<dyn Iterator<Item = (Vec<Ix>, Self::LeafElement)> + 'a> {
        self.as_ref().indexed_iter_logical_order(prefix)
    }
}

impl<'a, T: 'a> ApproxAssertIterHelper<'a> for &'a [T]
where
    &'a T: ApproxAssertIterHelper<'a>,
{
    type LeafElement = <&'a T as ApproxAssertIterHelper<'a>>::LeafElement;

    fn indexed_iter_logical_order(
        self,
        prefix: Vec<Ix>,
    ) -> Box<dyn Iterator<Item = (Vec<Ix>, Self::LeafElement)> + 'a> {
        let iter = self.iter().enumerate().flat_map(move |(idx, el)| {
            let mut new_prefix = prefix.clone();
            new_prefix.push(idx);
            el.indexed_iter_logical_order(new_prefix)
        });

        Box::new(iter)
    }
}

impl<'a, S, D> ApproxAssertIterHelper<'a> for &'a ArrayBase<S, D>
where
    S: Data,
    S::Elem: Copy,
    &'a S::Elem: ApproxAssertIterHelper<'a>,
    D: Dimension,
{
    type LeafElement = S::Elem;

    fn indexed_iter_logical_order(
        self,
        prefix: Vec<Ix>,
    ) -> Box<dyn Iterator<Item = (Vec<Ix>, Self::LeafElement)> + 'a> {
        let iter = self.indexed_iter().map(move |(idx, elm)| {
            let mut new_prefix = prefix.clone();
            new_prefix.extend(idx.into_dimension().as_array_view().iter());
            (new_prefix, *elm)
        });

        Box::new(iter)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

    use ndarray::{arr1, arr2, arr3};

    use super::*;

    #[test]
    fn test_nan_safe_f32_cmp_sorts_in_the_right_order() {
        let data = &mut [f32::NAN, 1., 5., f32::NAN, 4.];
        data.sort_by(nan_safe_f32_cmp);

        assert_approx_eq!(f32, &data[2..], [1., 4., 5.], ulps = 0);
        assert!(data[0].is_nan());
        assert!(data[1].is_nan());

        data.sort_by(nan_safe_f32_cmp_desc);

        assert_approx_eq!(f32, &data[..3], [5., 4., 1.], ulps = 0);
        assert!(data[3].is_nan());
        assert!(data[4].is_nan());

        let data = &mut [1., 5., 3., 4.];

        data.sort_by(nan_safe_f32_cmp);
        assert_approx_eq!(f32, &data[..], [1., 3., 4., 5.], ulps = 0);

        data.sort_by(nan_safe_f32_cmp_desc);
        assert_approx_eq!(f32, &data[..], [5., 4., 3., 1.], ulps = 0);
    }

    #[test]
    fn test_nan_safe_f32_cmp_nans_compare_as_expected() {
        assert_eq!(nan_safe_f32_cmp(&f32::NAN, &f32::NAN), Ordering::Equal);
        assert_eq!(nan_safe_f32_cmp(&-12., &f32::NAN), Ordering::Greater);
        assert_eq!(nan_safe_f32_cmp_desc(&-12., &f32::NAN), Ordering::Less);
        assert_eq!(nan_safe_f32_cmp(&f32::NAN, &-12.), Ordering::Less);
        assert_eq!(nan_safe_f32_cmp_desc(&f32::NAN, &-12.), Ordering::Greater);
        assert_eq!(nan_safe_f32_cmp(&12., &f32::NAN), Ordering::Greater);
        assert_eq!(nan_safe_f32_cmp_desc(&12., &f32::NAN), Ordering::Less);
        assert_eq!(nan_safe_f32_cmp(&f32::NAN, &12.), Ordering::Less);
        assert_eq!(nan_safe_f32_cmp_desc(&f32::NAN, &12.), Ordering::Greater);
    }

    #[test]
    fn test_assert_approx_eq_float() {
        assert_approx_eq!(f32, 0.15039155, 0.1503916, ulps = 3);
        catch_unwind(|| assert_approx_eq!(f32, 0.15039155, 0.1503916, ulps = 2)).unwrap_err();
    }

    #[test]
    fn test_assert_approx_eq_iterable_1d() {
        assert_approx_eq!(f32, &[0.25, 1.25], &[0.25, 1.25]);
        assert_approx_eq!(f32, &[0.25, 1.25], arr1(&[0.25, 1.25]));
    }

    #[test]
    #[should_panic(expected = "at index [1]")]
    fn test_assert_approx_eq_fails() {
        assert_approx_eq!(f32, &[0.35, 4.35], arr1(&[0.35, 4.45]));
    }

    #[test]
    #[should_panic(expected = "at index [0, 1, 2]")]
    fn test_assert_approx_eq_fails_multi_dimensional() {
        assert_approx_eq!(
            f32,
            &[[[0.25, 1.25, 0.], [0.0, 0.125, 0.]]],
            arr3(&[[[0.25, 1.25, 0.], [0.0, 0.125, 1.]]])
        );
    }

    #[test]
    fn test_assert_approx_eq_iterable_nested() {
        assert_approx_eq!(
            f32,
            &[[0.25, 1.25], [0.0, 0.125]],
            &[[0.25, 1.25], [0.0, 0.125]]
        );
        assert_approx_eq!(
            f32,
            &[[0.25, 1.25], [0.0, 0.125]],
            arr2(&[[0.25, 1.25], [0.0, 0.125]])
        );
        assert_approx_eq!(
            f32,
            &[[[0.25, 1.25], [0.0, 0.125]]],
            arr3(&[[[0.25, 1.25], [0.0, 0.125]]])
        );
    }

    #[test]
    #[should_panic(expected = "[0, 2]")]
    fn test_panic_at_different_length() {
        assert_approx_eq!(f32, &[[1., 2., 3.]], &[[1., 2.]]);
    }
}
