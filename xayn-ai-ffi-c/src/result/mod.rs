pub(crate) mod error;
pub(crate) mod fault;

use std::panic::{catch_unwind, RefUnwindSafe, UnwindSafe};

use ffi_support::{ExternError, IntoFfi};

/// Calls a callback which returns a result.
///
/// Similar to [`ffi_support::call_with_result()`] but with additional functionality:
/// - Ok: returns `T`'s FFI value.
/// - Error: returns `T`'s default FFI value and optionally reports an error.
/// - Panic: returns `T`'s default FFI value, performs cleanup and optionally reports an error.
pub fn call_with_result<F, G, T>(call: F, clean: G, error: Option<&mut ExternError>) -> T::Value
where
    F: UnwindSafe + FnOnce() -> Result<T, ExternError>,
    G: RefUnwindSafe + FnOnce(),
    T: IntoFfi,
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
                *error = cause;
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
