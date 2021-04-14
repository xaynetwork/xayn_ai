use std::panic::{catch_unwind, RefUnwindSafe, UnwindSafe};

use ffi_support::{ExternError, IntoFfi};

/// This function does nothing.
///
/// Calling this prevents Swift to drop the library.
#[no_mangle]
pub extern "C" fn dummy_function() {}

/// Calls a callback which returns a result.
///
/// Similar to [`ffi_support::call_with_result()`] but with additional functionality:
/// - `Ok(T)`: returns `T`'s FFI value.
/// - `Error(E)`: returns `T`'s default FFI value and optionally reports an error.
/// - Panic: returns `T`'s default FFI value, performs cleanup and optionally reports an error.
pub fn call_with_result<F, G, T, E>(call: F, clean: G, error: Option<&mut ExternError>) -> T::Value
where
    F: UnwindSafe + FnOnce() -> Result<T, E>,
    G: RefUnwindSafe + FnOnce(),
    T: IntoFfi,
    E: Into<ExternError>,
{
    match catch_unwind(call) {
        Ok(Ok(value)) => {
            if let Some(error) = error {
                *error = ExternError::success();
            }
            value.into_ffi_value()
        }
        Ok(Err(cause)) => {
            if let Some(error) = error {
                *error = cause.into();
            }
            T::ffi_default()
        }
        Err(cause) => {
            if let Some(error) = error {
                *error = cause.into();
            }
            clean();
            T::ffi_default()
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    /// Common casts from references to pointers.
    ///
    /// By default, a im/mutable reference to `Self` is cast as a im/mutable pointer to `Self`. In
    /// addition, the target type `T` can be changed as well.
    ///
    /// # Safety
    /// The cast itself is safe, although it is unsafe to use the resulting pointer. The behavior is
    /// undefined if:
    /// - A `T` different from `Self` doesn't have the same memory layout.
    /// - A pointer is accessed after the lifetime of the corresponding reference ends.
    /// - A pointer of an immutable reference is accessed mutably.
    pub trait AsPtr<'a, T = Self>
    where
        Self: 'a,
    {
        /// Casts the immutable reference as a constant pointer.
        #[inline]
        fn as_ptr(&self) -> *const T {
            self as *const Self as *const T
        }

        /// Casts the mutable reference as a mutable pointer.
        #[inline]
        fn as_mut_ptr(&mut self) -> *mut T {
            self as *mut Self as *mut T
        }
    }
}
