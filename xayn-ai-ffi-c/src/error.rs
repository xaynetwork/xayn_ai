use std::{
    cmp::Ordering,
    mem::transmute,
    ptr::{null, null_mut},
    slice::from_raw_parts_mut,
};

use displaydoc::Display;
use ffi_support::{destroy_c_string, ErrorCode, ExternError, IntoFfi};
use thiserror::Error;
use xayn_ai::Error as AiError;

use crate::utils::call_with_result;

/// The Xayn AI error codes.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Display, Error)]
pub enum CError {
    /// An uncritical error.
    Warning = -2,
    /// An irrecoverable error.
    Panic = -1,
    /// No error.
    Success = 0,
    /// A vocab null pointer error.
    VocabPointer = 1,
    /// A model null pointer error.
    ModelPointer = 2,
    /// A vocab or model file IO error.
    ReadFile = 3,
    /// A Xayn AI initialization error.
    InitAi = 4,
    /// A Xayn AI null pointer error.
    AiPointer = 5,
    /// A document history null pointer error.
    HistoryPointer = 6,
    /// A document history id null pointer error.
    HistoryIdPointer = 7,
    /// A documents null pointer error.
    DocumentsPointer = 8,
    /// A document id null pointer error.
    DocumentIdPointer = 9,
    /// A document snippet null pointer error.
    DocumentSnippetPointer = 14,
    /// Pointer is null but size > 0 or size == 0 but pointer is not null.
    SerializedPointer = 15,
    /// Deserialization of reranker data error.
    RerankerDeserialization = 16,
    /// Serialization of reranker data error.
    RerankerSerialization = 17,
    /// An internal error.
    Internal = 1024,
}

impl PartialEq<ErrorCode> for CError {
    fn eq(&self, other: &ErrorCode) -> bool {
        (*self as i32).eq(&other.code())
    }
}

impl PartialEq<CError> for ErrorCode {
    fn eq(&self, other: &CError) -> bool {
        other.eq(self)
    }
}

impl PartialOrd<ErrorCode> for CError {
    fn partial_cmp(&self, other: &ErrorCode) -> Option<Ordering> {
        (*self as i32).partial_cmp(&other.code())
    }
}

impl PartialOrd<CError> for ErrorCode {
    fn partial_cmp(&self, other: &CError) -> Option<Ordering> {
        other.partial_cmp(self)
    }
}

impl From<ErrorCode> for CError {
    fn from(code: ErrorCode) -> Self {
        if code < Self::Warning || code >= Self::Internal {
            Self::Internal
        } else {
            unsafe { transmute(code) }
        }
    }
}

impl CError {
    /// Provides context for the error code.
    pub fn with_context(self, message: impl Into<String>) -> ExternError {
        ExternError::new_error(ErrorCode::new(self as i32), message)
    }

    /// See [`error_message_drop()`] for more.
    unsafe fn drop_message(error: *const ExternError) {
        if let Some(error) = unsafe { error.as_ref() } {
            unsafe { destroy_c_string(error.get_raw_message() as *mut _) }
        }
    }
}

/// Frees the memory of the error message.
///
/// This *does not* free the error memory itself, which is allocated somewhere else. But this *does*
/// free the message field memory of the error. Not freeing the error message on consecutive errors
/// (ie. where the error code is not success) will potentially leak the error message memory of the
/// overwritten error.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
/// - A non-null error `message` doesn't point to memory allocated by [`xaynai_new()`] or
/// [`xaynai_rerank()`].
/// - A non-null error `message` is freed more than once.
/// - A non-null error `message` is accessed after being freed.
///
/// [`xaynai_new()`]: crate::ai::xaynai_new
/// [`xaynai_rerank()`]: crate::ai::xaynai_rerank
#[no_mangle]
pub unsafe extern "C" fn error_message_drop(error: *mut ExternError) {
    let drop = || {
        unsafe { CError::drop_message(error) };
        Result::<_, ExternError>::Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

/// The warnings (ie. uncritical errors) of the Xayn AI.
pub struct Warnings(Vec<ExternError>);

/// A raw slice of warnings.
#[repr(C)]
pub struct CWarnings {
    /// The raw pointer to the warnings.
    pub data: *const ExternError,
    /// The number of warnings.
    pub len: u32,
}

impl From<&[AiError]> for Warnings {
    fn from(warnings: &[AiError]) -> Self {
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
pub(crate) mod tests {
    use super::*;
    use crate::utils::tests::AsPtr;

    impl AsPtr<'_> for ExternError {}

    #[test]
    fn test_error() {
        assert_eq!(CError::Panic, ErrorCode::PANIC);
        assert_eq!(CError::Success, ErrorCode::SUCCESS);
    }
}
