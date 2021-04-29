//! Utilities.

use std::{ffi::CStr, fmt::Display, marker::PhantomData, mem, ptr, slice};

use crate::result::error::{CCode, Error};

/// This function does nothing.
///
/// Calling this prevents Swift to drop the library.
#[no_mangle]
pub extern "C" fn dummy_function() {}

/// Reads a string slice from the borrowed bytes pointer.
///
/// # Errors
/// Fails on null pointer and invalid utf8 encoding. The error is constructed from the `code`
/// and `context`.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `bytes` doesn't point to an aligned, contiguous area of memory with a terminating
/// null byte.
pub(crate) unsafe fn as_str<'a>(
    bytes: Option<&'a u8>,
    code: CCode,
    context: impl Display,
) -> Result<&'a str, Error> {
    let pointer = bytes
        .ok_or_else(|| code.with_context(format!("{}: The {} is null", context, code)))?
        as *const u8;
    unsafe { CStr::from_ptr::<'a>(pointer.cast()) }
        .to_str()
        .map_err(|cause| {
            code.with_context(format!(
                "{}: The {} contains invalid utf8: {}",
                context, code, cause,
            ))
        })
}

/// Conversion of Rust values into C-compatible values.
///
/// # Safety
/// The behavior is undefined if:
/// - The `Value` is not compatible with the C ABI.
/// - The `Value` is accessed after its lifetime has expired.
pub(crate) unsafe trait IntoRaw {
    /// A C-compatible value. Usually some kind of `#[repr(C)]` and `'static`/owned.
    // https://doc.rust-lang.org/std/boxed/index.html#memory-layout
    // https://rust-lang.github.io/unsafe-code-guidelines/layout/pointers.html
    type Value: Default + Send + Sized;

    /// Converts the Rust value into the C value. Usually leaks memory for heap allocated values.
    fn into_raw(self) -> Self::Value;
}

unsafe impl IntoRaw for () {
    // Safety: This is a no-op.
    type Value = ();

    #[inline]
    fn into_raw(self) -> Self::Value {}
}

#[repr(C)]
#[cfg_attr(test, derive(Debug))]
pub struct CBoxedSlice<T> {
    data: *const T,
    len: u64,
    _owned: PhantomData<T>,
}

// owned, unaliased
unsafe impl<T: Send> Send for CBoxedSlice<T> {}

impl<T> From<Box<[T]>> for CBoxedSlice<T> {
    fn from(boxed_slice: Box<[T]>) -> Self {
        let len = boxed_slice.len() as u64;
        let data = if boxed_slice.is_empty() {
            ptr::null()
        } else {
            Box::leak(boxed_slice).as_mut_ptr() as *const T
        };

        Self {
            data,
            len,
            _owned: PhantomData,
        }
    }
}

impl<T> Drop for CBoxedSlice<T> {
    fn drop(&mut self) {
        if self.is_sound() {
            let raw_slice = ptr::slice_from_raw_parts_mut(self.data as *mut T, self.len as usize);
            unsafe { Box::from_raw(raw_slice) };
        }
    }
}

impl<T> CBoxedSlice<T> {
    // pointer is non-null and aligend, slice is non-empty and addressable
    fn is_sound(&self) -> bool {
        !self.data.is_null()
            && self.data as usize % mem::align_of::<T>() == 0
            && self.len > 0
            && mem::size_of::<T>().saturating_mul(self.len as usize) <= isize::MAX as usize
    }

    pub fn as_slice(&self) -> &[T] {
        self.is_sound()
            .then(|| unsafe { slice::from_raw_parts(self.data, self.len as usize) })
            .unwrap_or_default()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.is_sound()
            .then(|| unsafe { slice::from_raw_parts_mut(self.data as *mut T, self.len as usize) })
            .unwrap_or_default()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Reads a string slice from the borrowed bytes pointer.
    ///
    /// # Panics
    /// Panics on null pointer and invalid utf8 encoding.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A non-null `bytes` doesn't point to an aligned, contiguous area of memory with a terminating
    /// null byte.
    pub fn as_str_unchecked<'a>(bytes: Option<&'a u8>) -> &'a str {
        unsafe { CStr::from_ptr::<'a>((bytes.unwrap() as *const u8).cast()) }
            .to_str()
            .unwrap()
    }

    impl CBoxedSlice<u8> {
        /// See [`as_str_unchecked()`] for more.
        pub fn as_str_unchecked(&self) -> &str {
            as_str_unchecked(self.as_slice().first())
        }
    }

    /// Nullable pointer conversions.
    pub trait AsPtr {
        /// Casts as a borrowed pointer.
        #[inline]
        fn as_ptr(&self) -> Option<&Self> {
            Some(self)
        }

        /// Casts as a mutable borrowed pointer.
        #[inline]
        fn as_mut_ptr(&mut self) -> Option<&mut Self> {
            Some(self)
        }

        /// Casts into an owned pointer.
        #[inline]
        fn into_ptr(self: Box<Self>) -> Option<Box<Self>> {
            Some(self)
        }
    }

    impl<T> AsPtr for CBoxedSlice<T> {}
}
