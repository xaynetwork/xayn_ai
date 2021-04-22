use ffi_support::{destroy_c_string, ErrorCode, ExternError};

use crate::result::call_with_result;

/// The Xayn AI error codes.
#[repr(i32)]
#[cfg_attr(test, derive(Clone, Copy, Debug))]
pub enum CCode {
    /// A warning or uncritical error.
    Fault = -2,
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
    /// A document histories null pointer error.
    HistoriesPointer = 6,
    /// A document history id null pointer error.
    HistoryIdPointer = 7,
    /// A documents null pointer error.
    DocumentsPointer = 8,
    /// A document id null pointer error.
    DocumentIdPointer = 9,
    /// A document snippet null pointer error.
    DocumentSnippetPointer = 10,
    /// Deserialization of reranker database error.
    RerankerDeserialization = 11,
    /// Serialization of reranker database error.
    RerankerSerialization = 12,
}

impl CCode {
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
/// - A non-null error `message` doesn't point to memory allocated by [`xaynai_new()`],
/// [`xaynai_rerank()`], [`xaynai_serialize()`], [`xaynai_faults()`] or [`xaynai_analytics()`].
/// - A non-null error `message` is freed more than once.
/// - A non-null error `message` is accessed after being freed.
///
/// [`xaynai_new()`]: crate::reranker::ai::xaynai_new
/// [`xaynai_rerank()`]: crate::reranker::ai::xaynai_rerank
/// [`xaynai_serialize()`]: crate::reranker::ai::xaynai_serialize
/// [`xaynai_faults()`]: crate::reranker::ai::xaynai_faults
/// [`xaynai_analytics()`]: crate::reranker::ai::xaynai_analytics
#[no_mangle]
pub unsafe extern "C" fn error_message_drop(error: *mut ExternError) {
    let drop = || {
        unsafe { CCode::drop_message(error) };
        Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::utils::tests::AsPtr;

    impl AsPtr for ExternError {}

    impl PartialEq<ErrorCode> for CCode {
        fn eq(&self, other: &ErrorCode) -> bool {
            (*self as i32).eq(&other.code())
        }
    }

    impl PartialEq<CCode> for ErrorCode {
        fn eq(&self, other: &CCode) -> bool {
            other.eq(self)
        }
    }

    #[test]
    fn test_error() {
        assert_eq!(CCode::Panic, ErrorCode::PANIC);
        assert_eq!(CCode::Success, ErrorCode::SUCCESS);
    }
}
