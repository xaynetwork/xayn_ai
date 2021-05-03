use std::{
    fmt,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
    slice,
};

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
/// - A null `data` pointer doesn't have `len` zero.
/// - A `len` is too large to address a corresponding `[T]`.
///
/// Also, it's undefined behavior to transfer ownership of boxed slice to Rust which wasn't
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
/// - The `data` doesn't own the memory or is accessed afterwards.
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
        let _owned = PhantomData;

        Self { data, len, _owned }
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
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
            .map(|data| {
                // Safety: Same as for `as_slice()`.
                unsafe { slice::from_raw_parts_mut(data.as_ptr(), self.len as usize) }
            })
            .unwrap_or_default()
    }

    /// Converts into a boxed slice.
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.data
            .map(|data| {
                // Safety: Same as for `drop()`.
                unsafe { into_boxed_slice_unchecked(data, self.len) }
            })
            .unwrap_or_default()
    }

    /// Gets the number of elements.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Checks for the presence of any elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> From<Box<[T]>> for CBoxedSlice<T> {
    fn from(boxed_slice: Box<[T]>) -> Self {
        Self::new(boxed_slice)
    }
}

impl<T> From<CBoxedSlice<T>> for Box<[T]> {
    fn from(boxed_slice: CBoxedSlice<T>) -> Self {
        boxed_slice.into_boxed_slice()
    }
}

impl<T: Clone> Clone for CBoxedSlice<T> {
    fn clone(&self) -> Self {
        self.as_slice().to_vec().into_boxed_slice().into()
    }
}

impl<T> Deref for CBoxedSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for CBoxedSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for CBoxedSlice<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for CBoxedSlice<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: fmt::Debug> fmt::Debug for CBoxedSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<T> fmt::Pointer for CBoxedSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(
            &self.data.map(NonNull::as_ptr).unwrap_or_else(ptr::null_mut),
            f,
        )
    }
}

// Safety: The data is owned and unaliased.
unsafe impl<T: Send> Send for CBoxedSlice<T> {}
unsafe impl<T: Sync> Sync for CBoxedSlice<T> {}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::utils::tests::{as_str_unchecked, AsPtr};

    impl CBoxedSlice<u8> {
        /// See [`as_str_unchecked()`] for more.
        pub fn as_str_unchecked(&self) -> &str {
            as_str_unchecked(self.as_slice().first())
        }
    }

    impl<T> AsPtr for CBoxedSlice<T> {}
}
