use std::panic::RefUnwindSafe;

use ffi_support::{implement_into_ffi_by_pointer, ExternError, FfiStr};
use xayn_ai::{Builder, Reranker};

use crate::{
    data::{
        document::CDocuments,
        history::CHistories,
        rank::{CRanks, Ranks},
    },
    reranker::{
        analytics::CAnalytics,
        bytes::{Bytes, CBytes},
    },
    result::{
        call_with_result,
        error::CCode,
        fault::{CFaults, Faults},
    },
};

/// The Xayn AI.
///
/// # Examples
/// - Create a Xayn AI with [`xaynai_new()`].
/// - Rerank documents with [`xaynai_rerank()`].
/// - Free memory with [`xaynai_drop()`], [`ranks_drop()`] and [`error_message_drop()`].
///
/// [`ranks_drop()`]: crate::data::rank::ranks_drop
/// [`error_message_drop()`]: crate::result::error::error_message_drop
pub struct CXaynAi(Reranker);

impl RefUnwindSafe for CXaynAi {
    // Safety:
    // The mutable fields `analytics`, `data` and `errors` of `CXaynAi.0` must not be accessed
    // after a panic if they could have been mutated. Currently, this can only happen in
    // `CXaynAi::rerank()` and `CXaynAi::drop()`.
    // Since we can't restore the last valid state after a panic without outside information, we
    // drop `CXaynAi` and signal a panic code.
}

implement_into_ffi_by_pointer! { CXaynAi }

impl CXaynAi {
    /// See [`xaynai_new()`] for more.
    unsafe fn new(
        vocab: FfiStr,
        model: FfiStr,
        serialized: *const CBytes,
    ) -> Result<Self, ExternError> {
        let vocab = vocab.as_opt_str().ok_or_else(|| {
            CCode::VocabPointer.with_context(
                "Failed to initialize the ai: The vocab is not a valid C-string pointer",
            )
        })?;
        let model = model.as_opt_str().ok_or_else(|| {
            CCode::ModelPointer.with_context(
                "Failed to initialize the ai: The model is not a valid C-string pointer",
            )
        })?;

        let serialized = unsafe { serialized.as_ref() }
            .map(|bytes| bytes.as_slice())
            .unwrap_or_default();

        Builder::default()
            .with_bert_from_file(vocab, model)
            .map_err(|cause| {
                CCode::ReadFile.with_context(format!("Failed to initialize the ai: {}", cause))
            })?
            .with_serialized_database(serialized)
            .map_err(|cause| {
                CCode::RerankerDeserialization.with_context(format!(
                    "Failed to deserialize the reranker database: {}",
                    cause,
                ))
            })?
            .build()
            .map(Self)
            .map_err(|cause| {
                CCode::InitAi.with_context(format!("Failed to initialize the ai: {}", cause))
            })
    }

    /// See [`xaynai_rerank()`] for more.
    unsafe fn rerank(
        xaynai: *mut Self,
        histories: *const CHistories,
        documents: *const CDocuments,
    ) -> Result<Ranks, ExternError> {
        let xaynai = unsafe { xaynai.as_mut() }.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to rerank the documents: The ai pointer is null")
        })?;

        let histories = unsafe { histories.as_ref() }
            .ok_or_else(|| {
                CCode::HistoriesPointer.with_context(
                    "Failed to rerank the documents: The document histories pointer is null",
                )
            })?
            .to_histories()?;
        let documents = unsafe { documents.as_ref() }
            .ok_or_else(|| {
                CCode::DocumentsPointer
                    .with_context("Failed to rerank the documents: The documents pointer is null")
            })?
            .to_documents()?;

        Ok(xaynai.0.rerank(&histories, &documents).into())
    }

    /// See [`xaynai_serialize()`] for more.
    unsafe fn serialize(xaynai: *const Self) -> Result<Bytes, ExternError> {
        let xaynai = unsafe { xaynai.as_ref() }.ok_or_else(|| {
            CCode::AiPointer
                .with_context("Failed to serialize the reranker database: The ai pointer is null")
        })?;

        xaynai.0.serialize().map(Bytes).map_err(|cause| {
            CCode::RerankerSerialization
                .with_context(format!("Failed to serialize the reranker: {}", cause))
        })
    }

    /// See [`xaynai_faults()`] for more.
    unsafe fn faults(xaynai: *const Self) -> Result<Faults, ExternError> {
        let xaynai = unsafe { xaynai.as_ref() }.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to get the faults: The ai pointer is null")
        })?;

        Ok(xaynai.0.errors().into())
    }

    /// See [`xaynai_analytics()`] for more.
    unsafe fn analytics(xaynai: *const Self) -> Result<CAnalytics, ExternError> {
        let xaynai = unsafe { xaynai.as_ref() }.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to get the analytics: The ai pointer is null")
        })?;

        Ok(CAnalytics(xaynai.0.analytics().cloned()))
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
/// Requires the vocabulary and model of the tokenizer/embedder. Optionally accepts the serialized
/// reranker database, otherwise creates a new one.
///
/// # Errors
/// Returns a null pointer if:
/// - The `vocab` or `model` paths are invalid.
/// - The `serialized` database is invalid.
/// - An unexpected panic happened.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `vocab` or `model` path doesn't point to an aligned, contiguous area of memory with
/// a terminating null byte.
/// - A non-null `serialized` database doesn't point to an aligned, contiguous area of memory.
/// - A serialized database `len` is too large to address the memory of a non-null serialized
/// database array.
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_new(
    vocab: FfiStr,
    model: FfiStr,
    serialized: *const CBytes,
    error: *mut ExternError,
) -> *mut CXaynAi {
    let new = || unsafe { CXaynAi::new(vocab, model, serialized) };
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
/// - An unexpected panic happened.
///
/// In case of a [`CCode::Panic`], the `xaynai` is dropped and must not be accessed anymore. The
/// last known valid state can be restored by the caller via [`xaynai_new()`] with a previously
/// serialized reranker database obtained from [`xaynai_serialize()`].
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
/// - A non-null `xaynai` is accessed after a panic.
///
/// [`CHistory`]: crate::data::history::CHistory
/// [`CDocument`]: crate::data::document::CDocument
#[no_mangle]
pub unsafe extern "C" fn xaynai_rerank(
    xaynai: *mut CXaynAi,
    histories: *const CHistories,
    documents: *const CDocuments,
    error: *mut ExternError,
) -> *mut CRanks {
    let rerank = || unsafe { CXaynAi::rerank(xaynai, histories, documents) };
    let clean = || unsafe { CXaynAi::drop(xaynai) };
    let error = unsafe { error.as_mut() };

    call_with_result(rerank, clean, error)
}

/// Serializes the database of the reranker.
///
/// # Errors
/// Returns a null pointer if:
/// - The xaynai is null.
/// - The serialization fails.
/// - An unexpected panic happened.
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
) -> *mut CBytes<'static> {
    let serialize = || unsafe { CXaynAi::serialize(xaynai) };
    let clean = || {};
    let error = unsafe { error.as_mut() };

    call_with_result(serialize, clean, error)
}

/// Retrieves faults which might occur during reranking.
///
/// Faults can range from warnings to errors which are handled in some default way internally.
///
/// # Errors
/// Returns a null pointer if:
/// - The `xaynai` is null.
/// - An unexpected panic happened.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `xaynai` doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_faults(
    xaynai: *mut CXaynAi,
    error: *mut ExternError,
) -> *mut CFaults {
    let faults = || unsafe { CXaynAi::faults(xaynai) };
    let clean = || {};
    let error = unsafe { error.as_mut() };

    call_with_result(faults, clean, error)
}

/// Retrieves the analytics which were collected in the penultimate reranking.
///
/// # Errors
/// Returns a null pointer if:
/// - The `xaynai` is null.
/// - An unexpected panic happened.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `xaynai` doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_analytics(
    xaynai: *mut CXaynAi,
    error: *mut ExternError,
) -> *mut CAnalytics {
    let analytics = || unsafe { CXaynAi::analytics(xaynai) };
    let clean = || {};
    let error = unsafe { error.as_mut() };

    call_with_result(analytics, clean, error)
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
        Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::CString,
        pin::Pin,
        ptr::{null, null_mut},
    };

    use super::*;
    use crate::{
        data::{document::tests::TestDocuments, history::tests::TestHistories, rank::ranks_drop},
        reranker::{analytics::analytics_drop, bytes::bytes_drop},
        result::{error::error_message_drop, fault::faults_drop},
        tests::{MODEL, VOCAB},
        utils::tests::AsPtr,
    };

    struct TestVocab(Pin<CString>);

    impl Drop for TestVocab {
        fn drop(&mut self) {}
    }

    impl TestVocab {
        fn as_ptr(&self) -> FfiStr {
            unsafe { FfiStr::from_raw(self.0.as_ptr()) }
        }
    }

    impl Default for TestVocab {
        fn default() -> Self {
            Self(Pin::new(CString::new(VOCAB).unwrap()))
        }
    }

    struct TestModel(Pin<CString>);

    impl Drop for TestModel {
        fn drop(&mut self) {}
    }

    impl TestModel {
        fn as_ptr(&self) -> FfiStr {
            unsafe { FfiStr::from_raw(self.0.as_ptr()) }
        }
    }

    impl Default for TestModel {
        fn default() -> Self {
            Self(Pin::new(CString::new(MODEL).unwrap()))
        }
    }

    pub struct TestDb<'a> {
        _vec: Pin<Vec<u8>>,
        bytes: CBytes<'a>,
    }

    impl Drop for TestDb<'_> {
        fn drop(&mut self) {}
    }

    impl<'a> AsPtr<CBytes<'a>> for TestDb<'a> {
        fn as_ptr(&self) -> *const CBytes<'a> {
            self.bytes.as_ptr()
        }

        fn as_mut_ptr(&mut self) -> *mut CBytes<'a> {
            self.bytes.as_mut_ptr()
        }
    }

    impl Default for TestDb<'_> {
        fn default() -> Self {
            let _vec = Pin::new(Vec::new());
            let bytes = _vec.as_ref().into();

            Self { _vec, bytes }
        }
    }

    #[test]
    fn test_rerank() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let hists = TestHistories::default();
        let docs = TestDocuments::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let xaynai = unsafe {
            xaynai_new(
                vocab.as_ptr(),
                model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CCode::Success);
        let ranks =
            unsafe { xaynai_rerank(xaynai, hists.as_ptr(), docs.as_ptr(), error.as_mut_ptr()) };
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { ranks_drop(ranks) };
        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_serialize() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let xaynai = unsafe {
            xaynai_new(
                vocab.as_ptr(),
                model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CCode::Success);
        let db = unsafe { xaynai_serialize(xaynai, error.as_mut_ptr()) };
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { bytes_drop(db) };
        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_faults() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let xaynai = unsafe {
            xaynai_new(
                vocab.as_ptr(),
                model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CCode::Success);
        let faults = unsafe { xaynai_faults(xaynai, error.as_mut_ptr()) };
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { faults_drop(faults) };
        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_analytics() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let xaynai = unsafe {
            xaynai_new(
                vocab.as_ptr(),
                model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CCode::Success);
        let analytics = unsafe { xaynai_analytics(xaynai, error.as_mut_ptr()) };
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { analytics_drop(analytics) };
        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_vocab_null() {
        let model = TestModel::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(
            unsafe { xaynai_new(invalid, model.as_ptr(), db.as_ptr(), error.as_mut_ptr()) }
                .is_null()
        );
        assert_eq!(error.get_code(), CCode::VocabPointer);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: The vocab is not a valid C-string pointer",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_vocab_invalid() {
        let model = TestModel::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let invalid = CString::new("").unwrap();
        let invalid = unsafe { FfiStr::from_raw(invalid.as_ptr()) };
        assert!(
            unsafe { xaynai_new(invalid, model.as_ptr(), db.as_ptr(), error.as_mut_ptr()) }
                .is_null()
        );
        assert_eq!(error.get_code(), CCode::ReadFile);
        assert!(error
            .get_message()
            .as_str()
            .contains("Failed to initialize the ai: Failed to load a data file: "));

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_model_null() {
        let vocab = TestVocab::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(
            unsafe { xaynai_new(vocab.as_ptr(), invalid, db.as_ptr(), error.as_mut_ptr()) }
                .is_null()
        );
        assert_eq!(error.get_code(), CCode::ModelPointer);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: The model is not a valid C-string pointer",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_model_invalid() {
        let vocab = TestVocab::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let invalid = CString::new("").unwrap();
        let invalid = unsafe { FfiStr::from_raw(invalid.as_ptr()) };
        assert!(
            unsafe { xaynai_new(vocab.as_ptr(), invalid, db.as_ptr(), error.as_mut_ptr()) }
                .is_null()
        );
        assert_eq!(error.get_code(), CCode::ReadFile);
        assert!(error
            .get_message()
            .as_str()
            .contains("Failed to initialize the ai: Failed to load a data file: "));

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
        assert_eq!(error.get_code(), CCode::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_serialize() {
        let mut error = ExternError::default();

        let invalid = null_mut();
        assert!(unsafe { xaynai_serialize(invalid, error.as_mut_ptr()) }.is_null());
        assert_eq!(error.get_code(), CCode::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to serialize the reranker database: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_faults() {
        let mut error = ExternError::default();

        let invalid = null_mut();
        assert!(unsafe { xaynai_faults(invalid, error.as_mut_ptr()) }.is_null());
        assert_eq!(error.get_code(), CCode::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to get the faults: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_analytics() {
        let mut error = ExternError::default();

        let invalid = null_mut();
        assert!(unsafe { xaynai_analytics(invalid, error.as_mut_ptr()) }.is_null());
        assert_eq!(error.get_code(), CCode::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to get the analytics: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_history_null() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let docs = TestDocuments::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let xaynai = unsafe {
            xaynai_new(
                vocab.as_ptr(),
                model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CCode::Success);

        let invalid = null();
        assert!(
            unsafe { xaynai_rerank(xaynai, invalid, docs.as_ptr(), error.as_mut_ptr(),) }.is_null()
        );
        assert_eq!(error.get_code(), CCode::HistoriesPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The document histories pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_documents_null() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let hists = TestHistories::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let xaynai = unsafe {
            xaynai_new(
                vocab.as_ptr(),
                model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CCode::Success);

        let invalid = null();
        assert!(
            unsafe { xaynai_rerank(xaynai, hists.as_ptr(), invalid, error.as_mut_ptr(),) }
                .is_null()
        );
        assert_eq!(error.get_code(), CCode::DocumentsPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The documents pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_serialized_empty() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let mut error = ExternError::default();

        let db = Pin::new(Vec::new());
        let db: CBytes = db.as_ref().into();
        let xaynai = unsafe {
            xaynai_new(
                vocab.as_ptr(),
                model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_serialized_invalid() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let mut error = ExternError::default();

        let version = u8::MAX;
        let invalid = Pin::new(vec![version]);
        let invalid: CBytes = invalid.as_ref().into();
        let xaynai = unsafe {
            xaynai_new(
                vocab.as_ptr(),
                model.as_ptr(),
                invalid.as_ptr(),
                error.as_mut_ptr(),
            )
        };
        assert!(xaynai.is_null());
        assert_eq!(error.get_code(), CCode::RerankerDeserialization);
        assert_eq!(
            error.get_message(),
            format!(
                "Failed to deserialize the reranker database: Unsupported serialized data. Found version {} expected 0",
                version,
            ).as_str(),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }
}
