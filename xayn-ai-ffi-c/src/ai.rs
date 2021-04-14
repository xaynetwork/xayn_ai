use std::{panic::RefUnwindSafe, slice};

use ffi_support::{implement_into_ffi_by_pointer, ExternError, FfiStr};
use xayn_ai::{Builder, Reranker};

use crate::{
    bytes::CBytes,
    doc::{
        document::CDocuments,
        history::CHistories,
        rank::{CRanks, Ranks},
    },
    result::{
        call_with_result,
        error::CError,
        warning::{CWarnings, Warnings},
    },
};

/// The Xayn AI.
///
/// # Examples
/// - Create a Xayn AI with [`xaynai_new()`].
/// - Rerank documents with [`xaynai_rerank()`].
/// - Free memory with [`xaynai_drop()`], [`ranks_drop()`] and [`error_message_drop()`].
///
/// [`ranks_drop()`]: crate::doc::rank::ranks_drop
/// [`error_message_drop()`]: crate::result::error::error_message_drop
pub struct CXaynAi(Reranker);

impl RefUnwindSafe for CXaynAi {
    // Safety:
    // The mutable fields `analytics`, `data` and `errors` of  `CXaynAi.0` must not be accessed
    // after a panic. We restore the last valid state after a panic without accessing those fields
    // and there is no direct access to them for a caller of the ffi.
}

implement_into_ffi_by_pointer! { CXaynAi }

impl CXaynAi {
    /// See [`xaynai_new()`] for more.
    unsafe fn new(
        serialized: *const u8,
        serialized_size: u32,
        vocab: FfiStr,
        model: FfiStr,
    ) -> Result<CXaynAi, ExternError> {
        if !(serialized.is_null() ^ (serialized_size > 0)) {
            return Err(
            CError::SerializedPointer.with_context(
                "Failed to initialize the ai: invalid combination of serialized and serialized_size",
            ));
        }
        let serialized = if serialized.is_null() {
            &[]
        } else {
            unsafe { slice::from_raw_parts(serialized, serialized_size as usize) }
        };
        let vocab = vocab.as_opt_str().ok_or_else(|| {
            CError::VocabPointer.with_context(
                "Failed to initialize the ai: The vocab is not a valid C-string pointer",
            )
        })?;
        let model = model.as_opt_str().ok_or_else(|| {
            CError::ModelPointer.with_context(
                "Failed to initialize the ai: The model is not a valid C-string pointer",
            )
        })?;

        Builder::default()
            .with_serialized_database(serialized)
            .map_err(|cause| {
                CError::RerankerDeserialization
                    .with_context(format!("Failed to deserialize reranker data: {}", cause))
            })?
            .with_bert_from_file(vocab, model)
            .map_err(|cause| {
                CError::ReadFile.with_context(format!("Failed to initialize the ai: {}", cause))
            })?
            .build()
            .map(CXaynAi)
            .map_err(|cause| {
                CError::InitAi.with_context(format!("Failed to initialize the ai: {}", cause))
            })
    }

    /// See [`xaynai_rerank()`] for more.
    unsafe fn rerank(
        xaynai: *mut Self,
        histories: *const CHistories,
        documents: *const CDocuments,
    ) -> Result<Ranks, ExternError> {
        let xaynai = unsafe { xaynai.as_mut() }.ok_or_else(|| {
            CError::AiPointer.with_context("Failed to rerank the documents: The ai pointer is null")
        })?;

        let histories = unsafe { histories.as_ref() }
            .ok_or_else(|| {
                CError::HistoryPointer.with_context(
                    "Failed to rerank the documents: The document histories pointer is null",
                )
            })?
            .to_histories()?;
        let documents = unsafe { documents.as_ref() }
            .ok_or_else(|| {
                CError::DocumentsPointer
                    .with_context("Failed to rerank the documents: The documents pointer is null")
            })?
            .to_documents()?;

        let ranks = xaynai.0.rerank(&histories, &documents);
        Ranks::from_reranked_documents(ranks, &documents)
    }

    /// See [`xaynai_serialize()`] for more.
    unsafe fn serialize(xaynai: *mut CXaynAi) -> Result<CBytes, ExternError> {
        let xaynai = unsafe { xaynai.as_mut() }.ok_or_else(|| {
            CError::AiPointer.with_context("Failed to rerank the documents: The ai pointer is null")
        })?;

        let bytes = xaynai.0.serialize().map_err(|cause| {
            CError::RerankerSerialization
                .with_context(format!("Failed to serialize reranker data: {}", cause))
        })?;

        Ok(CBytes::from_vec(bytes))
    }

    /// See [`xaynai_warnings()`] for more.
    unsafe fn warnings(xaynai: *mut Self) -> Result<Warnings, ExternError> {
        let xaynai = unsafe { xaynai.as_mut() }.ok_or_else(|| {
            CError::AiPointer.with_context("Failed to get the warnings: The ai pointer is null")
        })?;

        Ok(xaynai.0.errors().into())
    }

    /// Cleans the mutable parts of the state.
    ///
    /// This *must* be called in case of a panic to uphold the contract of `RefUnwindSafe`.
    unsafe fn clean(xaynai: *mut Self) {
        if let Some(xaynai) = unsafe { xaynai.as_mut() } {
            xaynai.0.reload();
        }
    }

    /// See [`xaynai_drop()`] for more.
    unsafe fn drop(xaynai: *mut Self) {
        if !xaynai.is_null() {
            unsafe { Box::from_raw(xaynai) };
        }
    }
}

/// Creates and initializes the Xayn AI.
///
/// # Errors
/// Returns a null pointer if:
/// - The `vocab` or `model` paths are invalid.
/// - The `serialized` database is invalid.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `serialized` database doesn't point to an aligned, contiguous area of memory.
/// - A serialized database `size` is too large to address the memory of a non-null serialized
/// database array.
/// - A non-null `vocab` or `model` path doesn't point to an aligned, contiguous area of memory with
/// a terminating null byte.
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_new(
    serialized: *const u8,
    serialized_size: u32,
    vocab: FfiStr,
    model: FfiStr,
    error: *mut ExternError,
) -> *mut CXaynAi {
    let new = || unsafe { CXaynAi::new(serialized, serialized_size, vocab, model) };
    let clean = || {};
    let error = unsafe { error.as_mut() };

    call_with_result(new, clean, error)
}

/// Reranks the documents with the Xayn AI.
///
/// # Errors
/// Returns a null pointer if:
/// - The `xaynai` is null.
/// - The document `histories` are invalid.
/// - The `documents` are invalid.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `xaynai` doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null `histories` doesn't point to an aligned, contiguous are of memory with a
/// [`CHistories`].
/// - A non-null histories `data` doesn't point to an aligned, contiguous area of memory with
/// at least histories `len` many [`CHistory`]s.
/// - A histories `len` is too large to address the memory of a non-null [`CHistory`] array.
/// - A non-null `documents` doesn't point to an aligned, contiguous area of memory with a
/// [`CDocuments`].
/// - A non-null documents `data` doesn't point to an aligned, contiguous area of memory with
/// at least documents `len` many [`CDocument`]s.
/// - A documents `len` is too large to address the memory of a non-null [`CDocument`] array.
/// - A non-null `id` or `snippet` doesn't point to an aligned, contiguous area of memory with a
/// terminating null byte.
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
/// - A non-null, zero-sized `ranks` array is dereferenced.
///
/// [`CHistory`]: crate::doc::history::CHistory
/// [`CDocument`]: crate::doc::document::CDocument
#[no_mangle]
pub unsafe extern "C" fn xaynai_rerank(
    xaynai: *mut CXaynAi,
    histories: *const CHistories,
    documents: *const CDocuments,
    error: *mut ExternError,
) -> *mut CRanks {
    let rerank = || unsafe { CXaynAi::rerank(xaynai, histories, documents) };
    let clean = || unsafe { CXaynAi::clean(xaynai) };
    let error = unsafe { error.as_mut() };

    call_with_result(rerank, clean, error)
}

/// Serializes the current state of the reranker.
///
/// # Errors
/// Returns a null pointer if:
/// - The xaynai is null.
/// - The serialization fails.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null error doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_serialize(
    xaynai: *mut CXaynAi,
    error: *mut ExternError,
) -> *mut CBytes {
    let serialize = || unsafe { CXaynAi::serialize(xaynai) };
    let clean = || unsafe { CXaynAi::clean(xaynai) };
    let error = unsafe { error.as_mut() };

    call_with_result(serialize, clean, error)
}

/// Retrieves warnings which might occur during reranking.
///
/// # Errors
/// Returns a null pointer if:
/// - The `xaynai` is null.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `xaynai` doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_warnings(
    xaynai: *mut CXaynAi,
    error: *mut ExternError,
) -> *mut CWarnings {
    let warnings = || unsafe { CXaynAi::warnings(xaynai) };
    let clean = || unsafe { CXaynAi::clean(xaynai) };
    let error = unsafe { error.as_mut() };

    call_with_result(warnings, clean, error)
}

/// Frees the memory of the Xayn AI.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `xaynai` doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null `xaynai` is freed more than once.
/// - A non-null `xaynai` is accessed after being freed.
#[no_mangle]
pub unsafe extern "C" fn xaynai_drop(xaynai: *mut CXaynAi) {
    let drop = || {
        unsafe { CXaynAi::drop(xaynai) };
        Result::<_, ExternError>::Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::CString,
        marker::PhantomData,
        pin::Pin,
        ptr::{null, null_mut},
    };

    use super::*;
    use crate::{
        doc::{document::tests::TestDocuments, history::tests::TestHistories, rank::ranks_drop},
        result::{error::error_message_drop, warning::warnings_drop},
        tests::{MODEL, VOCAB},
        utils::tests::AsPtr,
    };

    #[allow(dead_code)]
    struct TestFiles<'a, 'b, 'c>
    where
        'b: 'a,
        'c: 'a,
    {
        vocab: Pin<CString>,
        v: FfiStr<'a>,
        model: Pin<CString>,
        m: FfiStr<'b>,
        _variance: PhantomData<&'c (FfiStr<'a>, FfiStr<'b>)>,
    }

    impl Default for TestFiles<'_, '_, '_> {
        fn default() -> Self {
            let vocab = Pin::new(CString::new(VOCAB).unwrap());
            let v = unsafe { FfiStr::from_raw(vocab.as_ptr()) };

            let model = Pin::new(CString::new(MODEL).unwrap());
            let m = unsafe { FfiStr::from_raw(model.as_ptr()) };

            Self {
                vocab,
                v,
                model,
                m,
                _variance: PhantomData,
            }
        }
    }

    #[test]
    fn test_rerank() {
        let files = TestFiles::default();
        let hists = TestHistories::default();
        let docs = TestDocuments::default();
        let mut error = ExternError::default();

        let xaynai = unsafe { xaynai_new(null(), 0, files.v, files.m, error.as_mut_ptr()) };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CError::Success);
        let ranks =
            unsafe { xaynai_rerank(xaynai, hists.as_ptr(), docs.as_ptr(), error.as_mut_ptr()) };
        assert_eq!(error.get_code(), CError::Success);
        let warnings = unsafe { xaynai_warnings(xaynai, error.as_mut_ptr()) };
        assert_eq!(error.get_code(), CError::Success);

        unsafe { xaynai_drop(xaynai) };
        unsafe { ranks_drop(ranks) };
        unsafe { warnings_drop(warnings) };
    }

    #[test]
    fn test_vocab_null() {
        let files = TestFiles::default();
        let mut error = ExternError::default();

        let invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(unsafe { xaynai_new(null(), 0, invalid, files.m, error.as_mut_ptr()) }.is_null());
        assert_eq!(error.get_code(), CError::VocabPointer);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: The vocab is not a valid C-string pointer",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_vocab_invalid() {
        let files = TestFiles::default();
        let mut error = ExternError::default();

        let invalid = CString::new("").unwrap();
        let invalid = unsafe { FfiStr::from_raw(invalid.as_ptr()) };
        assert!(unsafe { xaynai_new(null(), 0, invalid, files.m, error.as_mut_ptr()) }.is_null());
        assert_eq!(error.get_code(), CError::ReadFile);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: Failed to load a data file: No such file or directory (os error 2)",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_model_null() {
        let files = TestFiles::default();
        let mut error = ExternError::default();

        let invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(unsafe { xaynai_new(null(), 0, files.v, invalid, error.as_mut_ptr()) }.is_null());
        assert_eq!(error.get_code(), CError::ModelPointer);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: The model is not a valid C-string pointer",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_model_invalid() {
        let files = TestFiles::default();
        let mut error = ExternError::default();

        let invalid = CString::new("").unwrap();
        let invalid = unsafe { FfiStr::from_raw(invalid.as_ptr()) };
        assert!(unsafe { xaynai_new(null(), 0, files.v, invalid, error.as_mut_ptr()) }.is_null());
        assert_eq!(error.get_code(), CError::ReadFile);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: Failed to load a data file: No such file or directory (os error 2)",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_rerank() {
        let hists = TestHistories::default();
        let docs = TestDocuments::default();
        let mut error = ExternError::default();

        let invalid = null_mut();
        assert!(unsafe {
            xaynai_rerank(invalid, hists.as_ptr(), docs.as_ptr(), error.as_mut_ptr())
        }
        .is_null());
        assert_eq!(error.get_code(), CError::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_warnings() {
        let mut error = ExternError::default();

        let invalid = null_mut();
        assert!(unsafe { xaynai_warnings(invalid, error.as_mut_ptr()) }.is_null());
        assert_eq!(error.get_code(), CError::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to get the warnings: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_history_null() {
        let files = TestFiles::default();
        let docs = TestDocuments::default();
        let mut error = ExternError::default();

        let xaynai = unsafe { xaynai_new(null(), 0, files.v, files.m, error.as_mut_ptr()) };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CError::Success);

        let invalid = null();
        assert!(
            unsafe { xaynai_rerank(xaynai, invalid, docs.as_ptr(), error.as_mut_ptr(),) }.is_null()
        );
        assert_eq!(error.get_code(), CError::HistoryPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The document histories pointer is null",
        );

        unsafe { xaynai_drop(xaynai) };
        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_documents_null() {
        let files = TestFiles::default();
        let hists = TestHistories::default();
        let mut error = ExternError::default();

        let xaynai = unsafe { xaynai_new(null(), 0, files.v, files.m, error.as_mut_ptr()) };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CError::Success);

        let invalid = null();
        assert!(
            unsafe { xaynai_rerank(xaynai, hists.as_ptr(), invalid, error.as_mut_ptr(),) }
                .is_null()
        );
        assert_eq!(error.get_code(), CError::DocumentsPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The documents pointer is null",
        );

        unsafe { xaynai_drop(xaynai) };
        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_serialized_null_size_not_zero() {
        let files = TestFiles::default();
        let mut error = ExternError::default();

        let xaynai = unsafe { xaynai_new(null(), 1, files.v, files.m, error.as_mut_ptr()) };
        assert!(xaynai.is_null());
        assert_eq!(error.get_code(), CError::SerializedPointer);
    }

    #[test]
    fn test_serialized_not_null_size_zero() {
        let files = TestFiles::default();
        let mut error = ExternError::default();

        let serialized = vec![1u8];
        let xaynai =
            unsafe { xaynai_new(serialized.as_ptr(), 0, files.v, files.m, error.as_mut_ptr()) };
        assert!(xaynai.is_null());
        assert_eq!(error.get_code(), CError::SerializedPointer);
    }

    #[test]
    fn test_serialized_invalid() {
        let files = TestFiles::default();
        let mut error = ExternError::default();

        let serialized = vec![1u8];
        let xaynai = unsafe {
            xaynai_new(
                serialized.as_ptr(),
                serialized.len() as u32,
                files.v,
                files.m,
                error.as_mut_ptr(),
            )
        };
        assert!(xaynai.is_null());
        assert_eq!(error.get_code(), CError::RerankerDeserialization);
    }
}
