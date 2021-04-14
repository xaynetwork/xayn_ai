use std::{
    ptr::{null, null_mut},
    slice::from_raw_parts_mut,
};

use ffi_support::{ExternError, IntoFfi};
use xayn_ai::Error;

use crate::result::{call_with_result, error::CError};

/// The Xayn Ai warnings.
pub struct Warnings(Vec<ExternError>);

/// A raw slice of warnings.
#[repr(C)]
pub struct CWarnings {
    /// The raw pointer to the warnings.
    pub data: *const ExternError,
    /// The number of warnings.
    pub len: u32,
}

impl From<&[Error]> for Warnings {
    fn from(warnings: &[Error]) -> Self {
        Self(
            warnings
                .iter()
                .map(|warning| CError::Warning.with_context(format!("{}", warning)))
                .collect::<Vec<_>>(),
        )
    }
}

unsafe impl IntoFfi for Warnings {
    type Value = *mut CWarnings;

    #[inline]
    fn ffi_default() -> Self::Value {
        null_mut()
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        let len = self.0.len() as u32;
        let data = if self.0.is_empty() {
            null()
        } else {
            self.0.leak().as_ptr()
        };
        let warnings = CWarnings { data, len };

        Box::into_raw(Box::new(warnings))
    }
}

impl CWarnings {
    /// See [`warnings_drop()`] for more.
    unsafe fn drop(warnings: *mut Self) {
        if !warnings.is_null() {
            let warnings = unsafe { Box::from_raw(warnings) };
            if !warnings.data.is_null() && warnings.len > 0 {
                unsafe {
                    Box::from_raw(from_raw_parts_mut(
                        warnings.data as *mut ExternError,
                        warnings.len as usize,
                    ))
                };
            }
        }
    }
}

/// Frees the memory of the warnings.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `warnings` doesn't point to memory allocated by [`xaynai_warnings()`].
/// - A non-null `warnings` is freed more than once.
/// - A non-null `warnings` is accessed after being freed.
///
/// [`xaynai_warnings()`]: crate::ai::xaynai_warnings
#[no_mangle]
pub unsafe extern "C" fn warnings_drop(warnings: *mut CWarnings) {
    let drop = || unsafe {
        CWarnings::drop(warnings);
        Result::<_, ExternError>::Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[cfg(test)]
mod tests {
    use std::slice::from_raw_parts;

    use itertools::izip;

    use super::*;

    pub struct TestErrors(Vec<Error>);

    impl Default for TestErrors {
        fn default() -> Self {
            Self(vec![
                Error::msg("this is a warning"),
                Error::msg("and another warning"),
            ])
        }
    }

    #[test]
    fn test_from_error() {
        let errors = TestErrors::default();
        let warnings = Warnings::from(errors.0.as_slice());
        assert_eq!(warnings.0.len(), errors.0.len());
        for (warning, error) in izip!(warnings.0, errors.0) {
            assert_eq!(warning.get_code(), CError::Warning);
            assert_eq!(warning.get_message().as_str(), format!("{}", error));
        }
    }

    #[test]
    fn test_from_empty() {
        let warnings = Warnings::from(Vec::new().as_slice());
        assert!(warnings.0.is_empty());
    }

    #[test]
    fn test_into_raw() {
        let errors = TestErrors::default();
        let warnings = Warnings::from(errors.0.as_slice()).into_ffi_value();

        assert!(!warnings.is_null());
        let data = unsafe { &*warnings }.data;
        let len = unsafe { &*warnings }.len as usize;
        assert!(!data.is_null());
        assert_eq!(len, errors.0.len());
        for (warning, error) in izip!(unsafe { from_raw_parts(data, len) }, errors.0) {
            assert_eq!(warning.get_code(), CError::Warning);
            assert_eq!(warning.get_message().as_str(), format!("{}", error));
        }

        unsafe { warnings_drop(warnings) };
    }

    #[test]
    fn test_into_empty() {
        let warnings = Warnings(vec![]).into_ffi_value();

        assert!(!warnings.is_null());
        assert!(unsafe { &*warnings }.data.is_null());
        assert_eq!(unsafe { &*warnings }.len, 0);

        unsafe { warnings_drop(warnings) };
    }
}
