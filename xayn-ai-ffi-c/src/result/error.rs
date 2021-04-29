use std::{any::Any, convert::Infallible, ffi::CString, panic::AssertUnwindSafe};

use derive_more::Display;

use crate::{result::call_with_result, utils::IntoRaw};

/// The Xayn AI error codes.
#[repr(i8)]
#[derive(Clone, Copy, Display)]
#[cfg_attr(test, derive(Debug, PartialEq))]
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
    pub message: Option<Box<u8>>,
}

impl From<Infallible> for Error {
    fn from(_success: Infallible) -> Self {
        Self::success()
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
        let message = if let CCode::Success = self.code {
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
            // Safety:
            // Casting a Box<[u8]> to a Box<u8> is sound, but it leaks all values except the very
            // first one. Since the slice is terminated with a single null byte, we are able to
            // recover the length and reclaim the memory.
            Some(unsafe { Box::from_raw(bytes.leak().as_mut_ptr()) })
        };

        CError {
            code: self.code,
            message,
        }
    }
}

impl Error {
    /// Creates the error information for the success code.
    pub fn success() -> Self {
        CCode::Success.with_context(String::new())
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
        Error::success().into_raw()
    }
}

impl CError {
    /// See [`error_message_drop()`] for more.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) unsafe fn drop_message(error: Option<&mut Self>) -> Result<(), Infallible> {
        if let Some(error) = error {
            if let Some(message) = error.message.take() {
                // Safety:
                // Casting a Box<u8> to a CString is sound, if it originated from boxed slice with
                // a terminating null byte.
                unsafe { CString::from_raw(Box::into_raw(message).cast()) };
            }
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
    use crate::utils::tests::{as_str_unchecked, AsPtr};

    impl AsPtr for CError {}

    #[test]
    fn test_into_raw_success() {
        let error = Error::success().into_raw();
        assert_eq!(error.code, CCode::Success);
        assert!(error.message.is_none());

        let error = CCode::Success.with_context("test success").into_raw();
        assert_eq!(error.code, CCode::Success);
        assert!(error.message.is_none());
    }

    #[test]
    fn test_into_raw_error() {
        let code = CCode::AiPointer;
        let message = "test error";
        let mut error = code.with_context(message).into_raw();

        assert_eq!(error.code, code);
        assert_eq!(
            as_str_unchecked(error.message.as_ref().map(AsRef::as_ref)),
            message,
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_into_raw_panic() {
        let message = "test panic";
        let payload = catch_unwind(|| panic_any(message)).unwrap_err();
        let mut error = Error::panic(payload).into_raw();

        assert_eq!(error.code, CCode::Panic);
        assert_eq!(
            as_str_unchecked(error.message.as_ref().map(AsRef::as_ref)),
            message,
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }
}
