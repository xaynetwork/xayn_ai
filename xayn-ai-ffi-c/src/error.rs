use std::panic::catch_unwind;

use ffi_support::{destroy_c_string, ErrorCode, ExternError};

/// The Xayn AI error codes.
#[repr(i32)]
#[cfg_attr(test, derive(Clone, Copy, Debug))]
#[cfg_attr(not(test), allow(dead_code))]
pub enum CXaynAiError {
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
    DocumentSnippetPointer = 10,
    /// Pointer is null but size > 0 or size == 0 but pointer is not null.
    SerializedPointer = 11,
    /// Deserialization of reranker data error.
    RerankerDeserialization = 12,
    /// Serialization of reranker data error.
    RerankerSerialization = 13,
    /// An internal error.
    Internal = 1024,
}

impl CXaynAiError {
    /// Provides context for the error code.
    pub fn with_context(self, message: impl Into<String>) -> ExternError {
        ExternError::new_error(ErrorCode::new(self as i32), message)
    }

    unsafe fn drop_message(error: *mut ExternError) {
        if let Some(error) = error.as_mut() {
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
/// - A non-null error doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
/// - A non-null error message doesn't point to memory allocated by [`xaynai_new()`] or
/// [`xaynai_rerank()`].
/// - A non-null error message is freed more than once.
/// - A non-null error message is accessed after being freed.
///
/// [`xaynai_new()`]: crate::ai::xaynai_new
/// [`xaynai_rerank()`]: crate::ai::xaynai_rerank
#[no_mangle]
pub unsafe extern "C" fn error_message_drop(error: *mut ExternError) {
    let _ = catch_unwind(|| unsafe { CXaynAiError::drop_message(error) });
}

#[cfg(test)]
pub(crate) mod tests {
    use ffi_support::ErrorCode;

    use super::*;

    impl PartialEq<ErrorCode> for CXaynAiError {
        fn eq(&self, other: &ErrorCode) -> bool {
            (*self as i32).eq(&other.code())
        }
    }

    impl PartialEq<CXaynAiError> for ErrorCode {
        fn eq(&self, other: &CXaynAiError) -> bool {
            other.eq(self)
        }
    }

    #[test]
    fn test_error() {
        assert_eq!(CXaynAiError::Panic, ErrorCode::PANIC);
        assert_eq!(CXaynAiError::Success, ErrorCode::SUCCESS);
    }
}
