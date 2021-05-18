pub mod io;
mod softmax;

pub use softmax::*;

/// A python like axis index where negative values can be used to index from the end.
///
/// Do not use this for performance sensitive code, like iteratively indexing a large array.
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
    use std::panic::catch_unwind;

    use super::*;

    #[test]
    fn test_pylike_idx_panics_on_overflow() {
        catch_unwind(|| relative_index(0, usize::MAX)).unwrap_err();
        catch_unwind(|| relative_index(isize::MAX, 1)).unwrap_err();
    }

    #[test]
    fn test_pylike_idx_panics_on_out_of_bounds() {
        catch_unwind(|| relative_index(3, 3)).unwrap_err();
        catch_unwind(|| relative_index(10, 3)).unwrap_err();
        catch_unwind(|| relative_index(-4, 3)).unwrap_err();
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
