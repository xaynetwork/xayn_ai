//! Utilities.

use std::{ffi::CStr, fmt::Display};

use crate::result::error::{CCode, Error};

/// This function does nothing.
///
/// Calling this prevents Swift to drop the library.
#[no_mangle]
pub extern "C" fn dummy_function() {}

/// A raw C string.
#[repr(transparent)]
#[cfg_attr(test, derive(Debug))]
pub struct CStrPtr<'a>(pub(crate) Option<&'a u8>);

impl<'a> CStrPtr<'a> {
    /// Reads a string from the pointer.
    ///
    /// # Errors
    /// Fails on null pointer and invalid utf8 encoding. The error is constructed from the `code`
    /// and `context`.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A non-null pointer doesn't point to an aligned, contiguous area of memory with a terminating
    /// null byte.
    pub unsafe fn as_str(&self, code: CCode, context: impl Display) -> Result<&'a str, Error> {
        let pointer = self
            .0
            .ok_or_else(|| code.with_context(format!("{}: The {} is null", context, code)))?
            as *const u8;
        unsafe { CStr::from_ptr::<'a>(pointer.cast()) }
            .to_str()
            .map_err(|cause| {
                code.with_context(format!(
                    "{}: The {} contains invalid utf8: {}",
                    context, code, cause
                ))
            })
    }
}

/// Conversion of Rust values into C-compatible values.
///
/// # Safety
/// The behavior is undefined if:
/// - The `Value` is not compatible with the C ABI.
/// - The `Value` is accessed after its lifetime has expired.
pub(crate) unsafe trait IntoRaw {
    /// A C-compatible value. Usually some kind of `#[repr(C)]` and `'static`.
    type Value: Default + Send + Sized;

    /// Converts the Rust value into the C value. Usually leaks memory for heap allocated values.
    fn into_raw(self) -> Self::Value;
}

unsafe impl IntoRaw for () {
    type Value = ();

    #[inline]
    fn into_raw(self) -> Self::Value {
        // Safety: This is a no-op.
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::ffi::CStr;

    use super::*;

    impl<'a, S> From<S> for CStrPtr<'a>
    where
        S: AsRef<CStr>,
    {
        /// Wraps the [`CStr`] as a pointer.
        fn from(string: S) -> Self {
            let pointer = string.as_ref().as_ptr().cast::<u8>();
            Self(unsafe {
                // Safety: The pointer comes from a valid &CStr.
                pointer.as_ref()
            })
        }
    }

    impl<'a> CStrPtr<'a> {
        /// Creates a null pointer.
        pub fn null() -> Self {
            Self(None)
        }

        /// Reads a string from the pointer.
        ///
        /// # Panics
        /// Panics on null pointer and invalid utf8 encoding.
        ///
        /// # Safety
        /// The behavior is undefined if:
        /// - A non-null pointer doesn't point to an aligned, contiguous area of memory with a
        /// terminating null byte.
        pub fn as_str_unchecked(&self) -> &'a str {
            let pointer = self.0.unwrap() as *const u8;
            unsafe { CStr::from_ptr::<'a>(pointer.cast()) }
                .to_str()
                .unwrap()
        }
    }
}
