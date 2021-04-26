use std::panic::AssertUnwindSafe;

use ffi_support::{ExternError, IntoFfi};
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
    utils::ptr_to_str,
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

unsafe impl IntoFfi for CXaynAi {
    type Value = Option<&'static mut CXaynAi>;

    #[inline]
    fn ffi_default() -> Self::Value {
        None
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        Some(Box::leak(Box::new(self)))
    }
}

impl CXaynAi {
    /// See [`xaynai_new()`] for more.
    unsafe fn new(
        vocab: Option<&u8>,
        model: Option<&u8>,
        serialized: Option<&CBytes>,
    ) -> Result<Self, ExternError> {
        let vocab =
            unsafe { ptr_to_str(vocab, CCode::VocabPointer, "Failed to initialize the ai") }?;
        let model =
            unsafe { ptr_to_str(model, CCode::ModelPointer, "Failed to initialize the ai") }?;

        let serialized = serialized
            .map(|bytes| unsafe { bytes.as_slice() })
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
        xaynai: Option<&mut Self>,
        histories: Option<&CHistories>,
        documents: Option<&CDocuments>,
    ) -> Result<Ranks, ExternError> {
        let xaynai = xaynai.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to rerank the documents: The ai pointer is null")
        })?;

        let histories = histories
            .ok_or_else(|| {
                CCode::HistoriesPointer.with_context(
                    "Failed to rerank the documents: The document histories pointer is null",
                )
            })?
            .to_histories()?;
        let documents = documents
            .ok_or_else(|| {
                CCode::DocumentsPointer
                    .with_context("Failed to rerank the documents: The documents pointer is null")
            })?
            .to_documents()?;

        Ok(xaynai.0.rerank(&histories, &documents).into())
    }

    /// See [`xaynai_serialize()`] for more.
    unsafe fn serialize(xaynai: Option<&Self>) -> Result<Bytes, ExternError> {
        let xaynai = xaynai.ok_or_else(|| {
            CCode::AiPointer
                .with_context("Failed to serialize the reranker database: The ai pointer is null")
        })?;

        xaynai.0.serialize().map(Bytes).map_err(|cause| {
            CCode::RerankerSerialization
                .with_context(format!("Failed to serialize the reranker: {}", cause))
        })
    }

    /// See [`xaynai_faults()`] for more.
    unsafe fn faults(xaynai: Option<&Self>) -> Result<Faults, ExternError> {
        let xaynai = xaynai.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to get the faults: The ai pointer is null")
        })?;

        Ok(xaynai.0.errors().into())
    }

    /// See [`xaynai_analytics()`] for more.
    unsafe fn analytics(xaynai: Option<&Self>) -> Result<CAnalytics, ExternError> {
        let xaynai = xaynai.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to get the analytics: The ai pointer is null")
        })?;

        Ok(CAnalytics(xaynai.0.analytics().cloned()))
    }

    /// See [`xaynai_drop()`] for more.
    unsafe fn drop(xaynai: Option<&mut Self>) {
        if let Some(xaynai) = xaynai {
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
    vocab: Option<&u8>,
    model: Option<&u8>,
    serialized: Option<&CBytes>,
    error: Option<&mut ExternError>,
) -> Option<&'static mut CXaynAi> {
    let new = || unsafe { CXaynAi::new(vocab, model, serialized) };

    call_with_result(new, error)
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
/// In case of a [`CCode::Panic`], the `xaynai` must not be accessed anymore and should be dropped
/// via [`xaynai_drop()`]. The last known valid state can be restored by the caller via
/// [`xaynai_new()`] with a previously serialized reranker database obtained from
/// [`xaynai_serialize()`].
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
    xaynai: Option<&mut CXaynAi>,
    histories: Option<&CHistories>,
    documents: Option<&CDocuments>,
    error: Option<&mut ExternError>,
) -> Option<&'static mut CRanks<'static>> {
    let rerank = AssertUnwindSafe(
        // Safety: It's the caller's responsibility to clean up in case of a panic.
        || unsafe { CXaynAi::rerank(xaynai, histories, documents) },
    );

    call_with_result(rerank, error)
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
    xaynai: Option<&CXaynAi>,
    error: Option<&mut ExternError>,
) -> Option<&'static mut CBytes<'static>> {
    let serialize = AssertUnwindSafe(
        // Safety: The mutable memory is not accessed in this call.
        || unsafe { CXaynAi::serialize(xaynai) },
    );

    call_with_result(serialize, error)
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
    xaynai: Option<&CXaynAi>,
    error: Option<&mut ExternError>,
) -> Option<&'static mut CFaults<'static>> {
    let faults = AssertUnwindSafe(
        // Safety: The mutable memory is not accessed in this call.
        || unsafe { CXaynAi::faults(xaynai) },
    );

    call_with_result(faults, error)
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
    xaynai: Option<&CXaynAi>,
    error: Option<&mut ExternError>,
) -> Option<&'static mut CAnalytics> {
    let analytics = AssertUnwindSafe(
        // Safety: The mutable memory is not accessed in this call.
        || unsafe { CXaynAi::analytics(xaynai) },
    );

    call_with_result(analytics, error)
}

/// Frees the memory of the Xayn AI.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `xaynai` doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null `xaynai` is freed more than once.
/// - A non-null `xaynai` is accessed after being freed.
#[no_mangle]
pub unsafe extern "C" fn xaynai_drop(xaynai: Option<&mut CXaynAi>) {
    let drop = AssertUnwindSafe(
        // Safety: The memory is dropped anyways.
        || {
            unsafe { CXaynAi::drop(xaynai) };
            Ok(())
        },
    );
    let error = None;

    call_with_result(drop, error);
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, pin::Pin};

    use super::*;
    use crate::{
        data::{document::tests::TestDocuments, history::tests::TestHistories, rank::ranks_drop},
        reranker::{analytics::analytics_drop, bytes::bytes_drop},
        result::{error::error_message_drop, fault::faults_drop},
        tests::{MODEL, VOCAB},
    };

    struct TestVocab(Pin<CString>);

    impl Drop for TestVocab {
        fn drop(&mut self) {}
    }

    impl Default for TestVocab {
        fn default() -> Self {
            Self(Pin::new(CString::new(VOCAB).unwrap()))
        }
    }

    impl TestVocab {
        fn as_ptr(&self) -> Option<&u8> {
            unsafe { self.0.as_ptr().cast::<u8>().as_ref() }
        }
    }

    struct TestModel(Pin<CString>);

    impl Drop for TestModel {
        fn drop(&mut self) {}
    }

    impl Default for TestModel {
        fn default() -> Self {
            Self(Pin::new(CString::new(MODEL).unwrap()))
        }
    }

    impl TestModel {
        fn as_ptr(&self) -> Option<&u8> {
            unsafe { self.0.as_ptr().cast::<u8>().as_ref() }
        }
    }

    pub struct TestDb<'a> {
        _vec: Pin<Vec<u8>>,
        bytes: CBytes<'a>,
    }

    impl Drop for TestDb<'_> {
        fn drop(&mut self) {}
    }

    impl TestDb<'_> {
        #[allow(clippy::unnecessary_wraps)]
        fn as_ptr(&self) -> Option<&CBytes> {
            Some(&self.bytes)
        }
    }

    impl Default for TestDb<'_> {
        fn default() -> Self {
            let mut _vec = Pin::new(Vec::new());
            let bytes = CBytes { data: None, len: 0 };

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
                Some(&mut error),
            )
        }
        .unwrap();
        assert_eq!(error.get_code(), CCode::Success);
        let ranks = unsafe {
            xaynai_rerank(
                Some(xaynai),
                hists.as_ptr(),
                docs.as_ptr(),
                Some(&mut error),
            )
        };
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { ranks_drop(ranks) };
        unsafe { xaynai_drop(Some(xaynai)) };
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
                Some(&mut error),
            )
        }
        .unwrap();
        assert_eq!(error.get_code(), CCode::Success);
        let db = unsafe { xaynai_serialize(Some(xaynai), Some(&mut error)) };
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { bytes_drop(db) };
        unsafe { xaynai_drop(Some(xaynai)) };
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
                Some(&mut error),
            )
        }
        .unwrap();
        assert_eq!(error.get_code(), CCode::Success);
        let faults = unsafe { xaynai_faults(Some(xaynai), Some(&mut error)) };
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { faults_drop(faults) };
        unsafe { xaynai_drop(Some(xaynai)) };
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
                Some(&mut error),
            )
        }
        .unwrap();
        assert_eq!(error.get_code(), CCode::Success);
        let analytics = unsafe { xaynai_analytics(Some(xaynai), Some(&mut error)) };
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { analytics_drop(analytics) };
        unsafe { xaynai_drop(Some(xaynai)) };
    }

    #[test]
    fn test_vocab_null() {
        let model = TestModel::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let invalid = None;
        assert!(
            unsafe { xaynai_new(invalid, model.as_ptr(), db.as_ptr(), Some(&mut error)) }.is_none()
        );
        assert_eq!(error.get_code(), CCode::VocabPointer);
        assert_eq!(
            error.get_message(),
            format!(
                "Failed to initialize the ai: The {} is null",
                CCode::VocabPointer,
            )
            .as_str(),
        );

        unsafe { error_message_drop(Some(&mut error)) };
    }

    #[test]
    fn test_vocab_invalid() {
        let model = TestModel::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let invalid = CString::new("").unwrap();
        let invalid = Some(unsafe { &*(invalid.as_ptr() as *const _) });
        assert!(
            unsafe { xaynai_new(invalid, model.as_ptr(), db.as_ptr(), Some(&mut error)) }.is_none()
        );
        assert_eq!(error.get_code(), CCode::ReadFile);
        assert!(error
            .get_message()
            .as_str()
            .contains("Failed to initialize the ai: Failed to load a data file: "));

        unsafe { error_message_drop(Some(&mut error)) };
    }

    #[test]
    fn test_model_null() {
        let vocab = TestVocab::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let invalid = None;
        assert!(
            unsafe { xaynai_new(vocab.as_ptr(), invalid, db.as_ptr(), Some(&mut error)) }.is_none()
        );
        assert_eq!(error.get_code(), CCode::ModelPointer);
        assert_eq!(
            error.get_message(),
            format!(
                "Failed to initialize the ai: The {} is null",
                CCode::ModelPointer,
            )
            .as_str(),
        );

        unsafe { error_message_drop(Some(&mut error)) };
    }

    #[test]
    fn test_model_invalid() {
        let vocab = TestVocab::default();
        let db = TestDb::default();
        let mut error = ExternError::default();

        let invalid = CString::new("").unwrap();
        let invalid = Some(unsafe { &*(invalid.as_ptr() as *const _) });
        assert!(
            unsafe { xaynai_new(vocab.as_ptr(), invalid, db.as_ptr(), Some(&mut error)) }.is_none()
        );
        assert_eq!(error.get_code(), CCode::ReadFile);
        assert!(error
            .get_message()
            .as_str()
            .contains("Failed to initialize the ai: Failed to load a data file: "));

        unsafe { error_message_drop(Some(&mut error)) };
    }

    #[test]
    fn test_ai_null_rerank() {
        let hists = TestHistories::default();
        let docs = TestDocuments::default();
        let mut error = ExternError::default();

        let invalid = None;
        assert!(
            unsafe { xaynai_rerank(invalid, hists.as_ptr(), docs.as_ptr(), Some(&mut error)) }
                .is_none()
        );
        assert_eq!(error.get_code(), CCode::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The ai pointer is null",
        );

        unsafe { error_message_drop(Some(&mut error)) };
    }

    #[test]
    fn test_ai_null_serialize() {
        let mut error = ExternError::default();

        let invalid = None;
        assert!(unsafe { xaynai_serialize(invalid, Some(&mut error)) }.is_none());
        assert_eq!(error.get_code(), CCode::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to serialize the reranker database: The ai pointer is null",
        );

        unsafe { error_message_drop(Some(&mut error)) };
    }

    #[test]
    fn test_ai_null_faults() {
        let mut error = ExternError::default();

        let invalid = None;
        assert!(unsafe { xaynai_faults(invalid, Some(&mut error)) }.is_none());
        assert_eq!(error.get_code(), CCode::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to get the faults: The ai pointer is null",
        );

        unsafe { error_message_drop(Some(&mut error)) };
    }

    #[test]
    fn test_ai_null_analytics() {
        let mut error = ExternError::default();

        let invalid = None;
        assert!(unsafe { xaynai_analytics(invalid, Some(&mut error)) }.is_none());
        assert_eq!(error.get_code(), CCode::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to get the analytics: The ai pointer is null",
        );

        unsafe { error_message_drop(Some(&mut error)) };
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
                Some(&mut error),
            )
        }
        .unwrap();
        assert_eq!(error.get_code(), CCode::Success);

        let invalid = None;
        assert!(
            unsafe { xaynai_rerank(Some(xaynai), invalid, docs.as_ptr(), Some(&mut error)) }
                .is_none()
        );
        assert_eq!(error.get_code(), CCode::HistoriesPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The document histories pointer is null",
        );

        unsafe { error_message_drop(Some(&mut error)) };
        unsafe { xaynai_drop(Some(xaynai)) };
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
                Some(&mut error),
            )
        }
        .unwrap();
        assert_eq!(error.get_code(), CCode::Success);

        let invalid = None;
        assert!(
            unsafe { xaynai_rerank(Some(xaynai), hists.as_ptr(), invalid, Some(&mut error)) }
                .is_none()
        );
        assert_eq!(error.get_code(), CCode::DocumentsPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The documents pointer is null",
        );

        unsafe { error_message_drop(Some(&mut error)) };
        unsafe { xaynai_drop(Some(xaynai)) };
    }

    #[test]
    fn test_serialized_empty() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let mut error = ExternError::default();

        let empty = CBytes { data: None, len: 0 };
        let empty = Some(&empty);
        let xaynai =
            unsafe { xaynai_new(vocab.as_ptr(), model.as_ptr(), empty, Some(&mut error)) }.unwrap();
        assert_eq!(error.get_code(), CCode::Success);

        unsafe { xaynai_drop(Some(xaynai)) };
    }

    #[test]
    fn test_serialized_invalid() {
        let vocab = TestVocab::default();
        let model = TestModel::default();
        let mut error = ExternError::default();

        let version = u8::MAX;
        let invalid = Pin::new(vec![version]);
        let invalid = CBytes {
            data: unsafe { invalid.as_ptr().as_ref() },
            len: invalid.len() as u32,
        };
        let invalid = Some(&invalid);
        assert!(
            unsafe { xaynai_new(vocab.as_ptr(), model.as_ptr(), invalid, Some(&mut error)) }
                .is_none()
        );
        assert_eq!(error.get_code(), CCode::RerankerDeserialization);
        assert_eq!(
            error.get_message(),
            format!(
                "Failed to deserialize the reranker database: Unsupported serialized data. Found version {} expected 0",
                version,
            ).as_str(),
        );

        unsafe { error_message_drop(Some(&mut error)) };
    }
}
