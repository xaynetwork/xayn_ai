use ffi_support::{abort_on_panic::with_abort_on_panic, destroy_c_string, ExternError};

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
    /// A Bert builder error.
    BuildBert = 4,
    /// A Reranker builder error.
    BuildReranker = 5,
    /// A Xayn AI null pointer error.
    XaynAiPointer = 6,
    /// A documents null pointer error.
    DocumentsPointer = 7,
    /// A document id null pointer error.
    IdPointer = 8,
    /// A document snippet null pointer error.
    SnippetPointer = 9,
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
    with_abort_on_panic(|| {
        if let Some(error) = error.as_mut() {
            unsafe { destroy_c_string(error.get_raw_message() as *mut _) }
        }
    })
}
