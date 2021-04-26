//! Error handling types.

pub(crate) mod error;
pub(crate) mod fault;

use std::panic::{catch_unwind, UnwindSafe};

use ffi_support::{ExternError, IntoFfi};

pub use self::{
    error::{error_message_drop, CCode},
    fault::{faults_drop, CFaults},
};

/// Calls a callback which returns a result.
///
/// Similar to [`ffi_support::call_with_result()`] but with optional error handling:
/// - Ok: returns `T`'s FFI value.
/// - Error/Panic: returns `T`'s default FFI value and optionally reports an error.
pub(crate) fn call_with_result<F, T>(call: F, error: Option<&mut ExternError>) -> T::Value
where
    F: UnwindSafe + FnOnce() -> Result<T, ExternError>,
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
            T::ffi_default()
        }
    }
}
