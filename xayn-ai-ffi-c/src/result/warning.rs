use std::{
    ptr::{null, null_mut},
    slice::from_raw_parts_mut,
};

use ffi_support::{destroy_c_string, ExternError, IntoFfi};
use xayn_ai::Error;

use crate::result::{call_with_result, error::CCode};

/// The Xayn Ai warnings.
pub struct Warnings(Vec<String>);

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
        Self(warnings.iter().map(ToString::to_string).collect())
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
            self.0
                .into_iter()
                .map(|message| CCode::Warning.with_context(message))
                .collect::<Vec<_>>()
                .leak()
                .as_ptr()
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
                let warnings = unsafe {
                    Box::from_raw(from_raw_parts_mut(
                        warnings.data as *mut ExternError,
                        warnings.len as usize,
                    ))
                };
                for warning in warnings.iter() {
                    unsafe { destroy_c_string(warning.get_raw_message() as *mut _) }
                }
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
/// [`xaynai_warnings()`]: crate::reranker::ai::xaynai_warnings
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

    struct TestWarnings(Vec<Error>);

    impl Default for TestWarnings {
        fn default() -> Self {
            Self(
                (0..10)
                    .map(|idx| Error::msg(format!("warning {}", idx)))
                    .collect(),
            )
        }
    }

    #[test]
    fn test_from_warnings() {
        let buffer = TestWarnings::default().0;
        let warnings = Warnings::from(buffer.as_slice());
        assert_eq!(warnings.0.len(), buffer.len());
        for (warning, error) in izip!(warnings.0, buffer) {
            assert_eq!(warning, error.to_string());
        }
    }

    #[test]
    fn test_from_empty() {
        let warnings = Warnings::from(Vec::new().as_slice());
        assert!(warnings.0.is_empty());
    }

    #[test]
    fn test_into_raw() {
        let buffer = TestWarnings::default().0;
        let warnings = Warnings::from(buffer.as_slice()).into_ffi_value();

        assert!(!warnings.is_null());
        let data = unsafe { &*warnings }.data;
        let len = unsafe { &*warnings }.len as usize;
        assert!(!data.is_null());
        assert_eq!(len, buffer.len());
        for (warning, error) in izip!(unsafe { from_raw_parts(data, len) }, buffer) {
            assert_eq!(warning.get_code(), CCode::Warning);
            assert_eq!(warning.get_message(), error.to_string().as_str());
        }

        unsafe { warnings_drop(warnings) };
    }

    #[test]
    fn test_into_empty() {
        let warnings = Warnings(Vec::new()).into_ffi_value();

        assert!(!warnings.is_null());
        assert!(unsafe { &*warnings }.data.is_null());
        assert_eq!(unsafe { &*warnings }.len, 0);

        unsafe { warnings_drop(warnings) };
    }
}
