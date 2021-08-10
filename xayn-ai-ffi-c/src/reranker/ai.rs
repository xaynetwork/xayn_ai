use std::panic::AssertUnwindSafe;

use xayn_ai::{Builder, RerankMode, Reranker};
use xayn_ai_ffi::{CCode, Error};

use crate::{
    data::{
        document::CDocuments,
        history::CHistories,
        outcomes::{CRerankingOutcomes, RerankingOutcomes},
    },
    reranker::{
        analytics::{Analytics, CAnalytics},
        bytes::{Bytes, CBytes},
    },
    result::{
        call_with_result,
        error::CError,
        fault::{CFaults, Faults},
    },
    utils::{as_str, IntoRaw},
};

/// The Xayn AI.
///
/// # Examples
/// - Create a Xayn AI with [`xaynai_new()`].
/// - Rerank documents with [`xaynai_rerank()`].
/// - Free memory with [`xaynai_drop()`], [`reranking_outcomes_drop()`] and
/// [`error_message_drop()`].
///
/// [`reranking_outcomes_drop()`]: crate::data::outcomes::reranking_outcomes_drop
/// [`error_message_drop()`]: crate::result::error::error_message_drop
pub struct CXaynAi(Reranker);

unsafe impl IntoRaw for CXaynAi
where
    Self: Sized,
{
    // Safety:
    // CXaynAi is sized, hence Box<CXaynAi> is representable as a *mut CXaynAi and
    // Option<Box<CXaynAi>> is eligible for the nullable pointer optimization.
    type Value = Option<Box<Self>>;

    #[inline]
    fn into_raw(self) -> Self::Value {
        Some(Box::new(self))
    }
}

const FAIL_INIT_AI: &str = "Failed to initialize the ai";

impl CXaynAi {
    /// See [`xaynai_new()`] for more.
    unsafe fn new(
        smbert_vocab: Option<&u8>,
        smbert_model: Option<&u8>,
        qambert_vocab: Option<&u8>,
        qambert_model: Option<&u8>,
        ltr_model: Option<&u8>,
        serialized: Option<&CBytes>,
    ) -> Result<Self, Error> {
        let smbert_vocab =
            unsafe { as_str(smbert_vocab, CCode::SMBertVocabPointer, FAIL_INIT_AI) }?;
        let smbert_model =
            unsafe { as_str(smbert_model, CCode::SMBertModelPointer, FAIL_INIT_AI) }?;
        let qambert_vocab =
            unsafe { as_str(qambert_vocab, CCode::QAMBertVocabPointer, FAIL_INIT_AI) }?;
        let qambert_model =
            unsafe { as_str(qambert_model, CCode::QAMBertModelPointer, FAIL_INIT_AI) }?;
        let ltr_model = unsafe { as_str(ltr_model, CCode::LtrModelPointer, FAIL_INIT_AI) }?;

        Builder::default()
            .with_smbert_from_file(smbert_vocab, smbert_model)
            .map_err(|cause| CCode::ReadFile.with_context(format!("{}: {}", FAIL_INIT_AI, cause)))?
            .with_qambert_from_file(qambert_vocab, qambert_model)
            .map_err(|cause| CCode::ReadFile.with_context(format!("{}: {}", FAIL_INIT_AI, cause)))?
            .with_domain_from_file(ltr_model)
            .map_err(|cause| CCode::ReadFile.with_context(format!("{}: {}", FAIL_INIT_AI, cause)))?
            .with_serialized_database(serialized)
            .map_err(|cause| {
                CCode::RerankerDeserialization.with_context(format!(
                    "Failed to deserialize the reranker database: {}",
                    cause,
                ))
            })?
            .build()
            .map(Self)
            .map_err(|cause| CCode::InitAi.with_context(format!("{}: {}", FAIL_INIT_AI, cause)))
    }

    /// See [`xaynai_rerank()`] for more.
    unsafe fn rerank(
        xaynai: Option<&mut Self>,
        mode: RerankMode,
        histories: Option<&CHistories>,
        documents: Option<&CDocuments>,
    ) -> Result<RerankingOutcomes, Error> {
        let xaynai = xaynai.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to rerank the documents: The ai pointer is null")
        })?;

        let histories = histories.ok_or_else(|| {
            CCode::HistoriesPointer.with_context(
                "Failed to rerank the documents: The document histories pointer is null",
            )
        })?;
        let histories = unsafe { histories.to_histories() }?;
        let documents = documents.ok_or_else(|| {
            CCode::DocumentsPointer
                .with_context("Failed to rerank the documents: The documents pointer is null")
        })?;
        let documents = unsafe { documents.to_documents() }?;

        Ok(xaynai.0.rerank(mode, &histories, &documents).into())
    }

    /// See [`xaynai_serialize()`] for more.
    unsafe fn serialize(xaynai: Option<&Self>) -> Result<Bytes, Error> {
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
    unsafe fn faults(xaynai: Option<&Self>) -> Result<Faults, Error> {
        let xaynai = xaynai.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to get the faults: The ai pointer is null")
        })?;

        Ok(xaynai.0.errors().into())
    }

    /// See [`xaynai_analytics()`] for more.
    unsafe fn analytics(xaynai: Option<&Self>) -> Result<Analytics, Error> {
        let xaynai = xaynai.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to get the analytics: The ai pointer is null")
        })?;

        Ok(Analytics(xaynai.0.analytics().cloned()))
    }

    /// See [`xaynai_syncdata_bytes()`] for more.
    unsafe fn syncdata_bytes(xaynai: Option<&Self>) -> Result<Bytes, Error> {
        let xaynai = xaynai.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to serialize sync data: The ai pointer is null")
        })?;

        xaynai.0.syncdata_bytes().map(Bytes).map_err(|cause| {
            CCode::SyncDataSerialization
                .with_context(format!("Failed to serialize sync data: {}", cause))
        })
    }

    /// See [`xaynai_synchronize()`] for more.
    unsafe fn synchronize(xaynai: Option<&mut Self>, bytes: Option<&CBytes>) -> Result<(), Error> {
        let xaynai = xaynai.ok_or_else(|| {
            CCode::AiPointer.with_context("Failed to synchronize data: The ai pointer is null")
        })?;

        let bytes = bytes.ok_or_else(|| {
            CCode::SyncDataBytesPointer
                .with_context("Failed to synchronize data: The bytes pointer is null")
        })?;

        xaynai.0.synchronize(bytes.as_ref()).map_err(|cause| {
            CCode::Synchronization.with_context(format!("Failed to synchronize data: {}", cause))
        })
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
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with a [`CError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_new(
    smbert_vocab: Option<&u8>,
    smbert_model: Option<&u8>,
    qambert_vocab: Option<&u8>,
    qambert_model: Option<&u8>,
    ltr_model: Option<&u8>,
    serialized: Option<&CBytes>,
    error: Option<&mut CError>,
) -> Option<Box<CXaynAi>> {
    let new = || unsafe {
        CXaynAi::new(
            smbert_vocab,
            smbert_model,
            qambert_vocab,
            qambert_model,
            ltr_model,
            serialized,
        )
    };

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
/// In case of a [`CCode::Panic`] the Xayn AI must not be accessed anymore and should be dropped via
/// [`xaynai_drop()`]. The last known valid state can be restored by the caller via [`xaynai_new()`]
/// with a previously serialized reranker database obtained from [`xaynai_serialize()`].
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `xaynai` doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null `histories` doesn't point to an aligned, contiguous are of memory with a
///   [`CHistories`] instance and all safety invariants of [`CHistories`] are uphold.
/// - A non-null `documents` doesn't point to an aligned, contiguous are of memory with a
///   [`CDocuments`] instance and all safety invariants of [`CDocuments`] are uphold.
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with a [`CError`].
/// - A non-null `xaynai` is accessed after a panic.
///
/// [`CHistory`]: crate::data::history::CHistory
/// [`CDocument`]: crate::data::document::CDocument
#[no_mangle]
pub unsafe extern "C" fn xaynai_rerank(
    xaynai: Option<&mut CXaynAi>,
    mode: RerankMode,
    histories: Option<&CHistories>,
    documents: Option<&CDocuments>,
    error: Option<&mut CError>,
) -> Option<Box<CRerankingOutcomes>> {
    let rerank = AssertUnwindSafe(
        // Safety: It's the caller's responsibility to clean up in case of a panic.
        || unsafe { CXaynAi::rerank(xaynai, mode, histories, documents) },
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
/// - A non-null error doesn't point to an aligned, contiguous area of memory with a [`CError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_serialize(
    xaynai: Option<&CXaynAi>,
    error: Option<&mut CError>,
) -> Option<Box<CBytes>> {
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
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with a [`CError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_faults(
    xaynai: Option<&CXaynAi>,
    error: Option<&mut CError>,
) -> Option<Box<CFaults>> {
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
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with a [`CError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_analytics(
    xaynai: Option<&CXaynAi>,
    error: Option<&mut CError>,
) -> Option<Box<CAnalytics>> {
    let analytics = AssertUnwindSafe(
        // Safety: The mutable memory is not accessed in this call.
        || unsafe { CXaynAi::analytics(xaynai) },
    );

    call_with_result(analytics, error)
}

/// Serializes the synchronizable data of the reranker.
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
/// - A non-null error doesn't point to an aligned, contiguous area of memory with a [`CError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_syncdata_bytes(
    xaynai: Option<&CXaynAi>,
    error: Option<&mut CError>,
) -> Option<Box<CBytes>> {
    let sync_bytes = AssertUnwindSafe(
        // Safety: The mutable memory is not accessed in this call.
        || unsafe { CXaynAi::syncdata_bytes(xaynai) },
    );

    call_with_result(sync_bytes, error)
}

/// Synchronizes the internal data of the reranker with another.
///
/// # Errors
/// - The `xaynai` is null.
/// - The serialized data `bytes` is null or invalid.
/// - The synchronization failed.
/// - An unexpected panic happened.
///
/// In case of a [`CCode::Panic`] the Xayn AI must not be accessed anymore and should be dropped via
/// [`xaynai_drop()`]. The last known valid state can be restored by the caller via [`xaynai_new()`]
/// with a previously serialized reranker database obtained from [`xaynai_serialize()`].
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `xaynai` doesn't point to memory allocated by [`xaynai_new()`].
/// - The safety constraints of [`CBoxedSlice`] are violated.
/// - A non-null `error` doesn't point to an aligned, contiguous area of memory with a [`CError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_synchronize(
    xaynai: Option<&mut CXaynAi>,
    bytes: Option<&CBytes>,
    error: Option<&mut CError>,
) {
    let synchronize = AssertUnwindSafe(|| unsafe { CXaynAi::synchronize(xaynai, bytes) });

    call_with_result(synchronize, error);
}

/// Frees the memory of the Xayn AI.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `xaynai` doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null `xaynai` is freed more than once.
/// - A non-null `xaynai` is accessed after being freed.
#[no_mangle]
pub unsafe extern "C" fn xaynai_drop(_xaynai: Option<Box<CXaynAi>>) {}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, marker::PhantomPinned, mem, path::PathBuf, pin::Pin};

    use tempfile::Builder as TempBuilder;

    use test_utils::{ltr, qambert, smbert};

    use super::*;
    use crate::{
        data::{
            document::tests::TestDocuments,
            history::tests::TestHistories,
            outcomes::reranking_outcomes_drop,
        },
        reranker::{analytics::analytics_drop, bytes::bytes_drop},
        result::{error::error_message_drop, fault::faults_drop},
        utils::tests::AsPtr,
    };

    impl AsPtr for CXaynAi {}

    struct TestFile<'a> {
        file: CString,
        ptr: Option<&'a u8>,
        _pinned: PhantomPinned,
    }

    impl<'a> TestFile<'a> {
        fn uninitialized(file: PathBuf) -> Pin<Box<Self>> {
            Box::pin(Self {
                file: CString::new(file.into_os_string().into_string().unwrap()).unwrap(),
                ptr: None,
                _pinned: PhantomPinned,
            })
        }

        fn initialize(mut self: Pin<Box<Self>>) -> Pin<Box<Self>> {
            let ptr = unsafe { self.file.as_ptr().cast::<u8>().as_ref() };
            unsafe { self.as_mut().get_unchecked_mut() }.ptr = ptr;

            self
        }

        fn smbert_vocab() -> Pin<Box<Self>> {
            Self::uninitialized(smbert::vocab().unwrap()).initialize()
        }

        fn smbert_model() -> Pin<Box<Self>> {
            Self::uninitialized(smbert::model().unwrap()).initialize()
        }

        fn qambert_vocab() -> Pin<Box<Self>> {
            Self::uninitialized(qambert::vocab().unwrap()).initialize()
        }

        fn qambert_model() -> Pin<Box<Self>> {
            Self::uninitialized(qambert::model().unwrap()).initialize()
        }

        fn ltr_model() -> Pin<Box<Self>> {
            Self::uninitialized(ltr::model().unwrap()).initialize()
        }

        #[allow(clippy::wrong_self_convention)] // false positive
        fn as_ptr(self: &'a Pin<Box<Self>>) -> Option<&'a u8> {
            self.ptr
        }
    }

    struct TestDb(CBytes);

    impl TestDb {
        #[allow(clippy::unnecessary_wraps)]
        fn as_ptr(&self) -> Option<&CBytes> {
            Some(&self.0)
        }
    }

    impl Default for TestDb {
        fn default() -> Self {
            Self(Vec::new().into_boxed_slice().into())
        }
    }

    struct TestSyncdata(CBytes);

    impl Default for TestSyncdata {
        fn default() -> Self {
            Self(vec![0; 17].into_boxed_slice().into())
        }
    }

    impl TestSyncdata {
        #[allow(clippy::unnecessary_wraps)]
        fn as_ptr(&self) -> Option<&CBytes> {
            Some(&self.0)
        }
    }

    /// Casts a function pointer to an extern C function to a unsafe extern C function which takes a pointer instead of a opt. box.
    ///
    /// The cast is safe (as far as possible), the usage of the resulting function not so much.
    fn cast_drop_fn<T>(func: unsafe extern "C" fn(Option<Box<T>>)) -> unsafe extern "C" fn(*mut T) {
        debug_assert_eq!(mem::size_of::<Option<Box<T>>>(), mem::size_of::<*mut T>());

        // We can't cast function pointers in rust using `as`.
        //
        // Safe: But due to non-null optimizations we know that `Option<Box<T>>` is equivalent to
        // `*mut T`, and as such `extern "C"` functions have the same layout. It should be noted
        // that this should only ever be used for testing, function pointers are one of the largely
        // under specified parts of rust, using "C" function pointers makes this better but I
        // wouldn't want to put this into release/production code. Still it helps us to better
        // test extern "C" destructors.
        unsafe { mem::transmute(func) }
    }

    #[test]
    fn test_rerank() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let hists = TestHistories::initialized();
        let docs = TestDocuments::initialized();
        let db = TestDb::default();
        let mut error = CError::default();

        let mut xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);
        let outcomes = unsafe {
            xaynai_rerank(
                xaynai.as_mut_ptr(),
                RerankMode::Search,
                hists.as_ptr(),
                docs.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);

        unsafe { cast_drop_fn(reranking_outcomes_drop)(Box::into_raw(outcomes)) };
        unsafe { cast_drop_fn(xaynai_drop)(Box::into_raw(xaynai)) };
    }

    #[test]
    fn test_serialize() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);
        let db = unsafe { xaynai_serialize(xaynai.as_ptr(), error.as_mut_ptr()) }.unwrap();
        assert_eq!(error.code, CCode::None);

        unsafe { bytes_drop(db.into_ptr()) };
        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_faults() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);
        let faults = unsafe { xaynai_faults(xaynai.as_ptr(), error.as_mut_ptr()) };
        assert_eq!(error.code, CCode::None);

        unsafe { faults_drop(faults) };
        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_analytics() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);
        // by default there are no analytics available
        let analytics = unsafe { xaynai_analytics(xaynai.as_ptr(), error.as_mut_ptr()) };
        assert!(analytics.is_none());
        assert_eq!(error.code, CCode::None);

        unsafe { analytics_drop(analytics) };
        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_syncdata_bytes() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);

        let syncdata = unsafe { xaynai_syncdata_bytes(xaynai.as_ptr(), error.as_mut_ptr()) };
        assert_eq!(error.code, CCode::None);
        assert!(syncdata.is_some());
        assert!(!syncdata.as_ref().unwrap().is_empty());

        unsafe { bytes_drop(syncdata) };
        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_synchronize() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let mut xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);

        let syncdata = TestSyncdata::default();
        unsafe { xaynai_synchronize(xaynai.as_mut_ptr(), syncdata.as_ptr(), error.as_mut_ptr()) };
        assert_eq!(error.code, CCode::None);

        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_smbert_vocab_null() {
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe {
            xaynai_new(
                invalid,
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::SMBertVocabPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
                "{}: The {} is null",
                FAIL_INIT_AI,
                CCode::SMBertVocabPointer,
            ),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_qambert_vocab_null() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                invalid,
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::QAMBertVocabPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
                "{}: The {} is null",
                FAIL_INIT_AI,
                CCode::QAMBertVocabPointer,
            ),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_smbert_vocab_empty() {
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = TempBuilder::new()
            .prefix("smbert_vocab")
            .suffix(".txt")
            .tempfile()
            .unwrap()
            .into_temp_path();
        let invalid = CString::new(invalid.to_str().unwrap()).unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                invalid,
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::InitAi);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
            "{}: Failed to build the tokenizer: Failed to build the tokenizer: Failed to build the model: Missing any entry in the vocabulary", FAIL_INIT_AI),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_qambert_vocab_empty() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = TempBuilder::new()
            .prefix("qambert_vocab")
            .suffix(".txt")
            .tempfile()
            .unwrap()
            .into_temp_path();
        let invalid = CString::new(invalid.to_str().unwrap()).unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                invalid,
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::InitAi);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
            "{}: Failed to build the tokenizer: Failed to build the tokenizer: Failed to build the model: Missing any entry in the vocabulary", FAIL_INIT_AI)
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_smbert_vocab_invalid() {
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = CString::new("").unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                invalid,
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::ReadFile);
        assert!(error
            .message
            .as_ref()
            .unwrap()
            .as_str()
            .contains(&format!("{}: Failed to load a data file: ", FAIL_INIT_AI)));

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_qambert_vocab_invalid() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = CString::new("").unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                invalid,
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::ReadFile);
        assert!(error
            .message
            .as_ref()
            .unwrap()
            .as_str()
            .contains(&format!("{}: Failed to load a data file: ", FAIL_INIT_AI)));

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_smbert_model_null() {
        let smbert_vocab = TestFile::smbert_vocab();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                invalid,
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::SMBertModelPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
                "{}: The {} is null",
                FAIL_INIT_AI,
                CCode::SMBertModelPointer,
            ),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_qambert_model_null() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                invalid,
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::QAMBertModelPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
                "{}: The {} is null",
                FAIL_INIT_AI,
                CCode::QAMBertModelPointer,
            ),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ltr_model_null() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                invalid,
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::LtrModelPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!("{}: The {} is null", FAIL_INIT_AI, CCode::LtrModelPointer,),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_smbert_model_empty() {
        let smbert_vocab = TestFile::smbert_vocab();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = TempBuilder::new()
            .prefix("smbert")
            .suffix(".onnx")
            .tempfile()
            .unwrap()
            .into_temp_path();
        let invalid = CString::new(invalid.to_str().unwrap()).unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                invalid,
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::InitAi);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
                "{}: Failed to build the model: Failed to run a tract operation: model proto does not contain a graph",
                FAIL_INIT_AI,
            ),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_qambert_model_empty() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = TempBuilder::new()
            .prefix("qambert")
            .suffix(".onnx")
            .tempfile()
            .unwrap()
            .into_temp_path();
        let invalid = CString::new(invalid.to_str().unwrap()).unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                invalid,
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::InitAi);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
                "{}: Failed to build the model: Failed to run a tract operation: model proto does not contain a graph",
                FAIL_INIT_AI,
            ),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ltr_model_empty() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = TempBuilder::new()
            .prefix("ltr")
            .suffix(".binparams")
            .tempfile()
            .unwrap()
            .into_temp_path();
        let invalid = CString::new(invalid.to_str().unwrap()).unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                invalid,
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::InitAi);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!("{}: io error: failed to fill whole buffer", FAIL_INIT_AI),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_smbert_model_invalid() {
        let smbert_vocab = TestFile::smbert_vocab();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = CString::new("").unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                invalid,
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::ReadFile);
        assert!(error
            .message
            .as_ref()
            .unwrap()
            .as_str()
            .contains(&format!("{}: Failed to load a data file: ", FAIL_INIT_AI)));

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_qambert_model_invalid() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = CString::new("").unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                invalid,
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::ReadFile);
        assert!(error
            .message
            .as_ref()
            .unwrap()
            .as_str()
            .contains(&format!("{}: Failed to load a data file: ", FAIL_INIT_AI)));

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ltr_model_invalid() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let invalid = CString::new("").unwrap();
        let invalid = unsafe { invalid.as_ptr().cast::<u8>().as_ref() };
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                invalid,
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::ReadFile);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!("{}: No such file or directory (os error 2)", FAIL_INIT_AI),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_rerank() {
        let hists = TestHistories::initialized();
        let docs = TestDocuments::initialized();
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe {
            xaynai_rerank(
                invalid,
                RerankMode::Search,
                hists.as_ptr(),
                docs.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::AiPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to rerank the documents: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_serialize() {
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe { xaynai_serialize(invalid, error.as_mut_ptr()) }.is_none());
        assert_eq!(error.code, CCode::AiPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to serialize the reranker database: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_faults() {
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe { xaynai_faults(invalid, error.as_mut_ptr()) }.is_none());
        assert_eq!(error.code, CCode::AiPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to get the faults: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_analytics() {
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe { xaynai_analytics(invalid, error.as_mut_ptr()) }.is_none());
        assert_eq!(error.code, CCode::AiPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to get the analytics: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_sync_bytes() {
        let mut error = CError::default();

        let invalid = None;
        assert!(unsafe { xaynai_syncdata_bytes(invalid, error.as_mut_ptr()) }.is_none());
        assert_eq!(error.code, CCode::AiPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to serialize sync data: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_ai_null_synchronize() {
        let mut error = CError::default();
        let syncdata = TestSyncdata::default();

        let null = None;
        unsafe { xaynai_synchronize(null, syncdata.as_ptr(), error.as_mut_ptr()) };
        assert_eq!(error.code, CCode::AiPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to synchronize data: The ai pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_history_null_rerank() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::smbert_vocab();
        let qambert_model = TestFile::smbert_model();
        let ltr_model = TestFile::ltr_model();
        let docs = TestDocuments::initialized();
        let db = TestDb::default();
        let mut error = CError::default();

        let mut xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);

        let invalid = None;
        assert!(unsafe {
            xaynai_rerank(
                xaynai.as_mut_ptr(),
                RerankMode::Search,
                invalid,
                docs.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::HistoriesPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to rerank the documents: The document histories pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_documents_null_rerank() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let hists = TestHistories::initialized();
        let db = TestDb::default();
        let mut error = CError::default();

        let mut xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);

        let invalid = None;
        assert!(unsafe {
            xaynai_rerank(
                xaynai.as_mut_ptr(),
                RerankMode::Search,
                hists.as_ptr(),
                invalid,
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::DocumentsPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to rerank the documents: The documents pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_serialized_empty() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let mut error = CError::default();

        let empty: CBytes = Vec::new().into_boxed_slice().into();
        let xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                empty.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);

        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_serialized_invalid() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let mut error = CError::default();

        let version = u8::MAX;
        let invalid = Bytes(vec![version]).into_raw().unwrap();
        assert!(unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                invalid.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .is_none());
        assert_eq!(error.code, CCode::RerankerDeserialization);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
                "Failed to deserialize the reranker database: Unsupported serialized data. Found version {} expected 1",
                version,
            ),
        );

        unsafe { bytes_drop(Some(invalid)) }
        unsafe { error_message_drop(error.as_mut_ptr()) };
    }

    #[test]
    fn test_bytes_null_synchronize() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let mut xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);

        let null = None;
        unsafe { xaynai_synchronize(xaynai.as_mut_ptr(), null, error.as_mut_ptr()) };
        assert_eq!(error.code, CCode::SyncDataBytesPointer);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to synchronize data: The bytes pointer is null",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_bytes_empty_synchronize() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let mut xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);

        let empty: CBytes = Vec::new().into_boxed_slice().into();
        unsafe { xaynai_synchronize(xaynai.as_mut_ptr(), Some(&empty), error.as_mut_ptr()) };
        assert_eq!(error.code, CCode::Synchronization);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            "Failed to synchronize data: Empty serialized data.",
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }

    #[test]
    fn test_bytes_invalid_synchronize() {
        let smbert_vocab = TestFile::smbert_vocab();
        let smbert_model = TestFile::smbert_model();
        let qambert_vocab = TestFile::qambert_vocab();
        let qambert_model = TestFile::qambert_model();
        let ltr_model = TestFile::ltr_model();
        let db = TestDb::default();
        let mut error = CError::default();

        let mut xaynai = unsafe {
            xaynai_new(
                smbert_vocab.as_ptr(),
                smbert_model.as_ptr(),
                qambert_vocab.as_ptr(),
                qambert_model.as_ptr(),
                ltr_model.as_ptr(),
                db.as_ptr(),
                error.as_mut_ptr(),
            )
        }
        .unwrap();
        assert_eq!(error.code, CCode::None);

        let version = u8::MAX;
        let invalid = Bytes(vec![version]).into_raw().unwrap();
        unsafe { xaynai_synchronize(xaynai.as_mut_ptr(), Some(&*invalid), error.as_mut_ptr()) };
        assert_eq!(error.code, CCode::Synchronization);
        assert_eq!(
            error.message.as_ref().unwrap().as_str(),
            format!(
                "Failed to synchronize data: Unsupported serialized data. Found version {} expected 0.",
                version,
            ),
        );

        unsafe { error_message_drop(error.as_mut_ptr()) };
        unsafe { xaynai_drop(xaynai.into_ptr()) };
    }
}
