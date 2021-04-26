//! Utilities.

use std::{ffi::CStr, fmt::Display};

use ffi_support::ExternError;

use crate::result::error::CCode;

/// This function does nothing.
///
/// Calling this prevents Swift to drop the library.
#[no_mangle]
pub extern "C" fn dummy_function() {}

pub(crate) unsafe fn ptr_to_str<'a>(
    pointer: Option<&'a u8>,
    code: CCode,
    prefix: impl Display,
) -> Result<&'a str, ExternError> {
    let pointer = pointer
        .ok_or_else(|| code.with_context(format!("{}: The {} is null", prefix, code)))?
        as *const u8;
    unsafe { CStr::from_ptr::<'a>(pointer.cast()) }
        .to_str()
        .map_err(|cause| {
            code.with_context(format!(
                "{}: The {} contains invalid utf8: {}",
                prefix, code, cause,
            ))
        })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::ffi::CStr;

    pub(crate) fn ptr_to_str_unchecked<'a>(pointer: Option<&'a u8>) -> &'a str {
        let pointer = pointer.unwrap() as *const u8;
        unsafe { CStr::from_ptr::<'a>(pointer.cast()) }
            .to_str()
            .unwrap()
    }
}
