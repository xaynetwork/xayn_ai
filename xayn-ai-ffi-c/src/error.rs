use std::{cmp::Ordering, mem::transmute, panic::catch_unwind};

use displaydoc::Display;
use ffi_support::{destroy_c_string, ErrorCode, ExternError};
use thiserror::Error;
use xayn_ai::Error;

use crate::utils::AsPtr;

impl AsPtr for ExternError {}

/// The Xayn AI error codes.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Display, Error)]
pub enum CError {
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
        if code < Self::Panic || code >= Self::Internal {
            Self::Internal
        } else {
            unsafe { transmute(code) }
        }
    }
}

impl CError {
    /// Provides extern context for the error code.
    pub fn with_extern_context(self, message: impl Into<String>) -> ExternError {
        ExternError::new_error(ErrorCode::new(self as i32), message)
    }

    /// Provides anyhow context for the error code.
    pub fn with_anyhow_context(self, message: Option<impl Into<String>>) -> Error {
        if let Some(message) = message {
            Error::new(self).context(message.into())
        } else {
            Error::new(self)
        }
    }

    /// See [`error_message_drop()`] for more.
    unsafe fn drop_message(error: *mut ExternError) {
        if let Some(error) = unsafe { error.as_mut() } {
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
    let _ = catch_unwind(|| unsafe { CError::drop_message(error) });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn test_error() {
        assert_eq!(CError::Panic, ErrorCode::PANIC);
        assert_eq!(CError::Success, ErrorCode::SUCCESS);
    }
}
