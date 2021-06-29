use std::{
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ptr::{self, NonNull},
    slice,
};

#[cfg(test)]
use ::{std::fmt, xayn_ai::ApproxAssertIterHelper};

/// A boxed slice with a C-compatible ABI.
///
/// # C Layout
/// ```
/// struct CBoxedSlice<T> {
///     data: *mut T,
///     len: u64,
/// }
/// ```
///
/// # Safety
/// If the boxed slice is only used within safe Rust or only immutably accessed outside of safe
/// Rust, everything is sound and effectively behaves like a `Box<[T]>`.
///
/// However, if it is mutably accessed outside of safe Rust, then it is undefined behavior if:
/// - A non-null `data` pointer doesn't point to an aligned, contiguous area of memory with exactly
/// `len` many `T`s.
/// - A null or dangling `data` pointer doesn't have `len` zero.
/// - A `len` is too large to address a corresponding `[T]`.
///
/// Also, it's undefined behavior to transfer ownership of a boxed slice to Rust which wasn't
/// allocated in Rust before.
///
/// A partial soundness check can be done via `is_sound()`, but this is of course only feasible to a
/// certain extend, ultimatly the caller is responsible to guarantee soundness when transferring
/// this over the FFI boundary.
#[repr(C)]
pub struct CBoxedSlice<T: Sized> {
    // behaves like a covariant *mut T
    data: Option<NonNull<T>>,
    // fixed width integer for bindgen compatibility, u64 to save overflow checks
    len: u64,
    // for dropcheck
    _owned: PhantomData<T>,
}

/// Creates a boxed slice from the pointer and length.
///
/// # Safety:
/// The behavior is undefined if:
/// - The `data` doesn't point to memory previously allocated in Rust as a `Box<[T]>` with the
/// corresponding `len`.
/// - The `data` pointer does own the memory and isn't used after wards in **any** way.
///   Ownership is moved into the returned `Box<[T]>`.
unsafe fn into_boxed_slice_unchecked<T: Sized>(data: NonNull<T>, len: u64) -> Box<[T]> {
    let raw_slice = ptr::slice_from_raw_parts_mut(data.as_ptr(), len as usize);
    unsafe { Box::from_raw(raw_slice) }
}

impl<T> Drop for CBoxedSlice<T> {
    fn drop(&mut self) {
        if self.is_sound() {
            if let Some(data) = self.data {
                // Safety:
                // The pointer is aligned and non-null and its underlying memory is addressable
                // within the length. The conversion of a valid pointer can't panic and the pointer
                // isn't accessed afterwards.
                // We can neither check that the pointer points to valid memory nor that the memory
                // was actually allocated in Rust as a `Box<[T]>` though. In case of usage within
                // safe Rust this is guaranteed, otherwise it's the caller's responsibility.
                unsafe { into_boxed_slice_unchecked(data, self.len) };
            }
        }
    }
}

impl<T> CBoxedSlice<T> {
    /// Creates a boxed slice.
    pub fn new(boxed_slice: Box<[T]>) -> Self {
        // Safety:
        // The conversion of a valid pointer can't panic. In case of a later panic, the memory is
        // freed via the `Drop` implementation.
        let len = boxed_slice.len() as u64;
        let data = NonNull::new(Box::leak(boxed_slice).as_mut_ptr());
        Self {
            data,
            len,
            _owned: PhantomData,
        }
    }

    // Creates an empty boxed slice with none hint.
    pub fn new_none() -> Self {
        Self {
            data: None,
            len: 0,
            _owned: PhantomData,
        }
    }

    /// Checks partially for soundness.
    ///
    /// This always holds if the boxed slice is only used within safe Rust. Otherwise this might be
    /// called to check:
    /// - Alignment of the pointer, even if it is dangling.
    /// - Addressability wrt. the current target pointer width.
    pub fn is_sound(&self) -> bool {
        if let Some(data) = self.data {
            data.as_ptr() as usize % mem::align_of::<T>() == 0
                && (mem::size_of::<T>() as u64).saturating_mul(self.len) <= isize::MAX as u64
        } else {
            self.len == 0
        }
    }

    /// Converts as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.data
            .map(|data| {
                // Safety:
                // The slice safety conditions must be ensured. In case of usage within safe Rust
                // this is guaranteed, otherwise it's the caller's responsibility.
                unsafe { slice::from_raw_parts(data.as_ptr() as *const T, self.len as usize) }
            })
            .unwrap_or_default()
    }

    /// Converts as a mutable slice.
    #[cfg_attr(not(doc), allow(dead_code))]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
            .map(|data| {
                // Safety: Same as for `as_slice()`.
                unsafe { slice::from_raw_parts_mut(data.as_ptr(), self.len as usize) }
            })
            .unwrap_or_default()
    }

    /// Converts into a boxed slice.
    #[cfg_attr(not(doc), allow(dead_code))]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        // Dropping it after calling into_boxed_slice_unchecked is
        // unsafe (use-after free) and if `data` is `None` we need
        // no drop even after into_boxed_slice_unchecked is called.
        let me = ManuallyDrop::new(self);
        me.data
            .map(|data| {
                // Safety: Same as for `drop()`.
                unsafe { into_boxed_slice_unchecked(data, me.len) }
            })
            .unwrap_or_default()
    }

    /// Gets the number of elements.
    #[cfg_attr(not(doc), allow(dead_code))]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Checks for the presence of any elements.
    #[cfg_attr(not(doc), allow(dead_code))]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Checks if it is not just empty but implies it is `None`.
    #[cfg_attr(not(doc), allow(dead_code))]
    pub fn is_none(&self) -> bool {
        self.data.is_none()
    }
}

#[cfg(test)]
impl<T: fmt::Debug> fmt::Debug for CBoxedSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<T> From<Vec<T>> for CBoxedSlice<T> {
    fn from(vec: Vec<T>) -> Self {
        Self::from(vec.into_boxed_slice())
    }
}

impl<T> From<Box<[T]>> for CBoxedSlice<T> {
    fn from(boxed_slice: Box<[T]>) -> Self {
        Self::new(boxed_slice)
    }
}

impl<T, E> From<Option<T>> for CBoxedSlice<E>
where
    CBoxedSlice<E>: From<T>,
{
    fn from(val: Option<T>) -> Self {
        if let Some(val) = val {
            Self::from(val)
        } else {
            Self::new_none()
        }
    }
}

impl<T> AsRef<[T]> for CBoxedSlice<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

#[cfg(test)]
impl<T1, T2, const N: usize> PartialEq<[T2; N]> for CBoxedSlice<T1>
where
    T1: PartialEq<T2>,
{
    fn eq(&self, other: &[T2; N]) -> bool {
        self.as_ref().eq(other)
    }
}

// Safety: The data is owned and unaliased.
unsafe impl<T: Send> Send for CBoxedSlice<T> {}
unsafe impl<T: Sync> Sync for CBoxedSlice<T> {}

#[cfg(test)]
impl<'a, T> ApproxAssertIterHelper<'a> for &'a CBoxedSlice<T>
where
    &'a T: ApproxAssertIterHelper<'a>,
{
    type LeafElement = <&'a T as ApproxAssertIterHelper<'a>>::LeafElement;

    fn indexed_iter_logical_order(
        self,
        prefix: Vec<usize>,
    ) -> Box<dyn Iterator<Item = (Vec<usize>, Self::LeafElement)> + 'a> {
        self.as_slice().indexed_iter_logical_order(prefix)
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::CStr, mem::ManuallyDrop};

    use super::*;
    use crate::utils::tests::AsPtr;

    impl CBoxedSlice<u8> {
        /// Interprets the slice as a &CStr and then &str.
        ///
        /// # Panic
        ///
        /// This panics if the bytes are not a valid string.
        pub fn as_str(&self) -> &str {
            CStr::from_bytes_with_nul(self.as_slice())
                .unwrap()
                .to_str()
                .unwrap()
        }
    }

    impl<T> AsPtr for CBoxedSlice<T> {}

    #[test]
    fn test_soundness_check() {
        let _owned = PhantomData::<f64>;
        let mut with_null_ptr = ManuallyDrop::new(CBoxedSlice {
            data: None,
            len: 0,
            _owned,
        });
        assert!(with_null_ptr.is_sound());
        with_null_ptr.len = 1;
        assert!(!with_null_ptr.is_sound());

        let mut with_dangeling = ManuallyDrop::new(CBoxedSlice {
            data: Some(NonNull::dangling()),
            len: 0,
            _owned,
        });
        assert!(with_dangeling.is_sound());
        // allocation soundness can't be checked by is sound
        with_dangeling.len = 4;
        assert!(with_dangeling.is_sound());

        with_dangeling.len = isize::MAX as u64;
        assert!(!with_dangeling.is_sound());

        with_dangeling.len = 0;
        assert!(with_dangeling.is_sound());
        with_dangeling.data = NonNull::new(1 as _);
        assert!(!with_dangeling.is_sound());
    }

    #[test]
    fn test_as_mut_slice() {
        let mut slice = CBoxedSlice::from(vec![1u8, 2, 4]);
        assert_eq!(slice.as_mut_slice(), &mut [1, 2, 4]);
    }

    #[test]
    fn test_into_boxed_slice() {
        let slice = CBoxedSlice::from(vec![1u8, 2, 4]);
        dbg!((&slice.data, &slice.len));
        dbg!(&slice);
        let slice = slice.into_boxed_slice();
        dbg!(&*slice as *const _);
        dbg!(&slice);
        assert_eq!(&*slice, &[1u8, 2, 4]);
    }

    #[test]
    fn test_len() {
        let slice = CBoxedSlice::from(vec![1u8, 2, 4]);
        assert_eq!(slice.len(), 3);
    }

    #[test]
    fn test_construct_from_option() {
        let slice = CBoxedSlice::from(Some(vec![1u8, 2, 4]));
        assert_eq!(slice.as_ref(), [1, 2, 4]);
        assert!(!slice.is_none());

        let none: Option<Vec<u8>> = None;
        let slice = CBoxedSlice::from(none);
        assert_eq!(slice.as_ref(), &[]);
        assert!(slice.is_none());
    }
}
