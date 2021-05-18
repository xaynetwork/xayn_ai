pub mod io;
mod softmax;

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

#[cfg(test)]
mod tests {
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
}
