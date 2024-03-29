//! Error handling types.

pub(crate) mod error;
pub(crate) mod fault;

use std::panic::{catch_unwind, UnwindSafe};

use xayn_ai_ffi::Error;

#[cfg(not(doc))]
use crate::result::error::CError;
use crate::utils::IntoRaw;

#[cfg(doc)]
pub use self::{
    error::{error_message_drop, CError},
    fault::{faults_drop, CFaults},
};

/// Calls a callback which returns a result.
///
/// Catches an unwinding panic with optional error handling:
/// - Ok: returns `T`'s FFI value.
/// - Error/Panic: returns `T`'s default FFI value and optionally reports an error.
pub(crate) fn call_with_result<F, T, E>(call: F, error: Option<&mut CError>) -> T::Value
where
    F: UnwindSafe + FnOnce() -> Result<T, E>,
    T: IntoRaw,
    E: Into<Error>,
{
    match catch_unwind(call) {
        Ok(Ok(value)) => {
            if let Some(error) = error {
                *error = Error::none().into_raw();
            }
            value.into_raw()
        }
        Ok(Err(cause)) => {
            if let Some(error) = error {
                *error = cause.into().into_raw();
            }
            T::Value::default()
        }
        Err(cause) => {
            if let Some(error) = error {
                *error = Error::panic(cause).into_raw();
            }
            T::Value::default()
        }
    }
}
