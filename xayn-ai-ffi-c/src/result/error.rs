use std::{any::Any, convert::Infallible, ffi::CString, panic::AssertUnwindSafe};

use derive_more::Display;

use crate::{reranker::CBytes, result::call_with_result, utils::IntoRaw};

/// The Xayn AI error codes.
#[repr(i8)]
#[derive(Clone, Copy, Display)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub enum CCode {
    /// A warning or noncritical error.
    Fault = -2,
    /// An irrecoverable error.
    Panic = -1,
    /// No error.
    None = 0,
    /// A smbert vocab null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    SMBertVocabPointer = 1,
    /// A smbert model null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    SMBertModelPointer = 2,
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
    /// A document session id null pointer error.
    DocumentSessionPointer = 11,
    DocumentQueryIdPointer = 12,
    DocumentQueryWordsPointer = 13,
    DocumentUrlPointer = 14,
    DocumentDomainPointer = 15,
    /// Deserialization of reranker database error.
    RerankerDeserialization = 16,
    /// Serialization of reranker database error.
    RerankerSerialization = 12,
    /// A qambert vocab null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    QAMBertVocabPointer = 13,
    /// A qambert model null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    QAMBertModelPointer = 14,
}

impl CCode {
    /// Provides context for the error code.
    pub fn with_context(self, message: impl Into<String>) -> Error {
        Error {
            code: self,
            message: message.into(),
        }
    }
}

/// The Xayn AI error information.
#[cfg_attr(test, derive(Debug))]
pub struct Error {
    pub code: CCode,
    pub message: String,
}

/// The raw Xayn AI error information.
#[repr(C)]
#[cfg_attr(test, derive(Debug))]
pub struct CError {
    /// The error code.
    pub code: CCode,
    /// The raw pointer to the error message.
    pub message: Option<Box<CBytes>>,
}

impl From<Infallible> for Error {
    fn from(_none: Infallible) -> Self {
        Self::none()
    }
}

unsafe impl IntoRaw for Error
where
    CError: Sized,
{
    // Safety: CError is sized and repr(C).
    type Value = CError;

    /// Creates the raw error information.
    ///
    /// If the code is success, then the message will be ignored, otherwise the message memory will
    /// be leaked. If the message contains null bytes, only the bytes up to the first null byte will
    /// be used.
    #[inline]
    fn into_raw(self) -> Self::Value {
        let message = if let CCode::None = self.code {
            None
        } else {
            let bytes = CString::new(self.message)
                .unwrap_or_else(|null| {
                    let position = null.nul_position();
                    CString::new(&null.into_vec()[..position]).unwrap(
                        // Safety: The bytes are cut off directly before the first null byte.
                    )
                })
                .into_bytes_with_nul();
            Some(Box::new(bytes.into_boxed_slice().into()))
        };

        CError {
            code: self.code,
            message,
        }
    }
}

impl Error {
    /// Creates the error information for the no error code.
    pub fn none() -> Self {
        CCode::None.with_context(String::new())
    }

    /// Creates the error information from the panic payload.
    pub fn panic(payload: Box<dyn Any + Send>) -> Self {
        // https://doc.rust-lang.org/std/panic/struct.PanicInfo.html#method.payload
        let message = if let Some(message) = payload.downcast_ref::<&str>() {
            message
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message
        } else {
            "Unknown panic"
        };

        CCode::Panic.with_context(message)
    }
}

impl Default for CError {
    /// Defaults to success.
    fn default() -> Self {
        Error::none().into_raw()
    }
}

impl CError {
    /// See [`error_message_drop()`] for more.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) unsafe fn drop_message(error: Option<&mut Self>) -> Result<(), Infallible> {
        if let Some(error) = error {
            error.message.take();
        }

        Ok(())
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
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with a [`CError`].
/// - A non-null error `message` doesn't point to memory allocated by [`xaynai_new()`],
/// [`xaynai_rerank()`], [`xaynai_serialize()`], [`xaynai_faults()`], [`xaynai_analytics()`] or
/// [`bytes_new()`].
/// - A non-null error `message` is freed more than once.
/// - A non-null error `message` is accessed after being freed.
///
/// [`xaynai_new()`]: crate::reranker::ai::xaynai_new
/// [`xaynai_rerank()`]: crate::reranker::ai::xaynai_rerank
/// [`xaynai_serialize()`]: crate::reranker::ai::xaynai_serialize
/// [`xaynai_faults()`]: crate::reranker::ai::xaynai_faults
/// [`xaynai_analytics()`]: crate::reranker::ai::xaynai_analytics
/// [`bytes_new()`]: crate::reranker::bytes::bytes_new
#[no_mangle]
pub unsafe extern "C" fn error_message_drop(error: Option<&mut CError>) {
    let drop = AssertUnwindSafe(
        // Safety: The memory is dropped anyways.
        || unsafe { CError::drop_message(error) },
    );
    let error = None;

    call_with_result(drop, error);
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, panic_any};

    use super::*;
    use crate::utils::tests::AsPtr;

    impl AsPtr for CError {}

    #[test]
    fn test_into_raw_success() {
        let error = Error::none().into_raw();
        assert_eq!(error.code, CCode::None);
        assert!(error.message.is_none());

        let error = CCode::None.with_context("test none").into_raw();
        assert_eq!(error.code, CCode::None);
        assert!(error.message.is_none());
    }

    #[test]
    fn test_into_raw_error() {
        let code = CCode::AiPointer;
        let message = "test error";
        let mut error = code.with_context(message).into_raw();

        assert_eq!(error.code, code);
        assert_eq!(error.message.as_ref().unwrap().as_str(), message);

        unsafe { error_message_drop(error.as_mut_ptr()) };
        assert!(error.message.is_none());
    }

    #[test]
    fn test_into_raw_panic() {
        let message = "test panic";
        let payload = catch_unwind(|| panic_any(message)).unwrap_err();
        let mut error = Error::panic(payload).into_raw();

        assert_eq!(error.code, CCode::Panic);
        assert_eq!(error.message.as_ref().unwrap().as_str(), message);

        unsafe { error_message_drop(error.as_mut_ptr()) };
        assert!(error.message.is_none());
    }
}
