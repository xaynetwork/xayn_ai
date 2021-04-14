use std::{
    collections::HashMap,
    marker::PhantomData,
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use ffi_support::{ExternError, FfiStr, IntoFfi};
use xayn_ai::{Document, DocumentHistory, DocumentsRank, Relevance, UserFeedback};

use crate::{error::CError, utils::call_with_result};

/// A document relevance level.
#[repr(u8)]
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub enum CRelevance {
    Low = 0,
    Medium = 1,
    High = 2,
}

impl From<CRelevance> for Relevance {
    fn from(relevance: CRelevance) -> Self {
        match relevance {
            CRelevance::Low => Self::Low,
            CRelevance::Medium => Self::Medium,
            CRelevance::High => Self::High,
        }
    }
}

/// A user feedback level.
#[repr(u8)]
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub enum CFeedback {
    Relevant = 0,
    Irrelevant = 1,
    None = 2,
}

impl From<CFeedback> for UserFeedback {
    fn from(feedback: CFeedback) -> Self {
        match feedback {
            CFeedback::Relevant => Self::Relevant,
            CFeedback::Irrelevant => Self::Irrelevant,
            CFeedback::None => Self::None,
        }
    }
}

/// A raw document history.
#[repr(C)]
pub struct CHistory<'a, 'b>
where
    'a: 'b,
{
    /// The raw pointer to the document id.
    pub id: FfiStr<'a>,
    /// The relevance level of the document.
    pub relevance: CRelevance,
    /// The user feedback level of the document.
    pub feedback: CFeedback,
    // covariant in lifetime and type
    _variance: PhantomData<&'b FfiStr<'a>>,
}

/// A raw slice of document histories.
#[repr(C)]
pub struct CHistories<'a, 'b, 'c>
where
    'a: 'b,
    'b: 'c,
{
    /// The raw pointer to the document histories.
    pub data: *const CHistory<'a, 'b>,
    /// The number of document histories.
    pub len: u32,
    // covariant in lifetime and type
    _variance: PhantomData<&'c [CHistory<'a, 'b>]>,
}

impl<'a, 'b, 'c> CHistories<'a, 'b, 'c>
where
    'a: 'b,
    'b: 'c,
{
    /// Collects the document histories from raw.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A non-null `data` doesn't point to an aligned, contiguous area of memory with at least
    /// `len` many [`CHistory`]s.
    /// - A `len` is too large to address the memory of a non-null [`CHistory`] array.
    /// - A non-null `id` doesn't point to an aligned, contiguous area of memory with a terminating
    /// null byte.
    pub unsafe fn to_histories(&self) -> Result<Vec<DocumentHistory>, ExternError> {
        if self.data.is_null() || self.len == 0 {
            return Ok(Vec::new());
        }

        unsafe { from_raw_parts(self.data, self.len as usize) }
            .iter()
            .map(|history| {
                let id = history
                    .id
                    .as_opt_str()
                    .map(Into::into)
                    .ok_or_else(|| {
                        CError::HistoryIdPointer.with_context(
                            "Failed to rerank the documents: A document history id is not a valid C-string pointer",
                        )
                    })?;
                let relevance = history.relevance.into();
                let user_feedback = history.feedback.into();

                Ok(DocumentHistory {id, relevance, user_feedback })
            })
            .collect()
    }
}

/// A raw document.
#[repr(C)]
pub struct CDocument<'a, 'b, 'c>
where
    'a: 'c,
    'b: 'c,
{
    /// The raw pointer to the document id.
    pub id: FfiStr<'a>,
    /// The raw pointer to the document snippet.
    pub snippet: FfiStr<'b>,
    /// The rank of the document.
    pub rank: u32,
    // covariant in lifetime and type
    _variance: PhantomData<&'c (FfiStr<'a>, FfiStr<'b>)>,
}

/// A raw slice of documents.
#[repr(C)]
pub struct CDocuments<'a, 'b, 'c, 'd>
where
    'a: 'c,
    'b: 'c,
    'c: 'd,
{
    /// The raw pointer to the documents.
    pub data: *const CDocument<'a, 'b, 'c>,
    /// The number of documents.
    pub len: u32,
    // covariant in lifetime and type
    _variance: PhantomData<&'d [CDocument<'a, 'b, 'c>]>,
}

impl<'a, 'b, 'c, 'd> CDocuments<'a, 'b, 'c, 'd>
where
    'a: 'c,
    'b: 'c,
    'c: 'd,
{
    /// Collects the documents from raw.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A non-null `data` doesn't point to an aligned, contiguous area of memory with at least
    /// `len` many [`CDocument`]s.
    /// - A `len` is too large to address the memory of a non-null [`CDocument`] array.
    /// - A non-null `id` or `snippet` doesn't point to an aligned, contiguous area of memory with a
    /// terminating null byte.
    pub unsafe fn to_documents(&self) -> Result<Vec<Document>, ExternError> {
        if self.data.is_null() || self.len == 0 {
            return Ok(Vec::new());
        }

        unsafe { from_raw_parts(self.data, self.len as usize) }
            .iter()
            .map(|document| {
                let id = document
                    .id
                    .as_opt_str()
                    .map(Into::into)
                    .ok_or_else(|| {
                        CError::DocumentIdPointer.with_context(
                            "Failed to rerank the documents: A document id is not a valid C-string pointer",
                        )
                    })?;
                let snippet = document
                    .snippet
                    .as_opt_str()
                    .map(Into::into)
                    .ok_or_else(|| {
                        CError::DocumentSnippetPointer.with_context(
                            "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
                        )
                    })?;
                let rank = document.rank as usize;

                Ok(Document { id, snippet, rank })
            })
            .collect()
    }
}

/// The ranks of the reranked documents.
///
/// The array is in the same order as the documents used in [`xaynai_rerank()`].
///
/// [`xaynai_rerank()`]: crate::ai::xaynai_rerank
pub struct CRanks(Vec<u32>);

unsafe impl IntoFfi for CRanks {
    type Value = *mut u32;

    #[inline]
    fn ffi_default() -> Self::Value {
        null_mut()
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        self.0.leak().as_mut_ptr()
    }
}

impl CRanks {
    /// Reorders the ranks wrt the documents.
    pub fn from_reranked_documents(
        ranks: DocumentsRank,
        documents: &[Document],
    ) -> Result<Self, ExternError> {
        let ranks = ranks
            .into_iter()
            .map(|(id, rank)| (id, rank as u32))
            .collect::<HashMap<_, _>>();
        documents
            .iter()
            .map(|document| ranks.get(&document.id).copied())
            .collect::<Option<Vec<_>>>()
            .map(Self)
            .ok_or_else(|| {
                CError::Internal.with_context(
                    "Failed to rerank the documents: The document ids are inconsistent",
                )
            })
    }

    /// See [`ranks_drop()`] for more.
    unsafe fn drop(ranks: *mut u32, len: u32) {
        if !ranks.is_null() && len > 0 {
            unsafe { Box::from_raw(from_raw_parts_mut(ranks, len as usize)) };
        }
    }
}

/// Frees the memory of the ranks array.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `ranks` doesn't point to memory allocated by [`xaynai_rerank()`].
/// - A non-zero `len` is different from the documents `len` used in [`xaynai_rerank()`].
/// - A non-null `ranks` is freed more than once.
/// - A non-null `ranks` is accessed after being freed.
///
/// [`xaynai_rerank()`]: crate::ai::xaynai_rerank
#[no_mangle]
pub unsafe extern "C" fn ranks_drop(ranks: *mut u32, len: u32) {
    let drop = || {
        unsafe { CRanks::drop(ranks, len) };
        Result::<_, ExternError>::Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[repr(C)]
pub struct CBytes {
    /// pointer to the data
    pub ptr: *const u8,
    /// number of bytes in the array
    pub len: u32,
}

unsafe impl IntoFfi for CBytes {
    type Value = *mut CBytes;

    #[inline]
    fn ffi_default() -> Self::Value {
        null_mut()
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        Box::into_raw(Box::new(self))
    }
}

impl CBytes {
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        if bytes.is_empty() {
            Self {
                ptr: null_mut(),
                len: 0,
            }
        } else {
            let len = bytes.len() as u32;
            let ptr = bytes.leak().as_mut_ptr();

            Self { ptr, len }
        }
    }

    fn drop(array: *mut CBytes) {
        if let Some(a) = unsafe { array.as_ref() } {
            if !a.ptr.is_null() && a.len > 0 {
                unsafe { Box::from_raw(from_raw_parts_mut(a.ptr as *mut u8, a.len as usize)) };
            }
            // Safety: we do not access `a` after we freed it
            unsafe { Box::from_raw(array) };
        }
    }
}

/// Frees the memory of a byte buffer.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null buffer doesn't point to memory allocated by [`xaynai_serialize()`].
/// - A non-null buffer is freed more than once.
/// - A non-null buffer is accessed after being freed.
///
/// [`xaynai_serialize()`]: crate::ai::xaynai_serialize
#[no_mangle]
pub unsafe extern "C" fn bytes_drop(buffer: *mut CBytes) {
    let drop = || {
        unsafe { CBytes::drop(buffer) };
        Result::<_, ExternError>::Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{ffi::CString, iter::repeat, pin::Pin, ptr::null};

    use itertools::izip;

    use super::*;
    use crate::utils::tests::AsPtr;

    #[allow(dead_code)]
    pub struct TestHistories<'a, 'b, 'c> {
        len: usize,
        ids: Pin<Vec<CString>>,
        history: Vec<CHistory<'a, 'b>>,
        histories: CHistories<'a, 'b, 'c>,
        _variance: PhantomData<&'c Pin<Vec<CString>>>,
    }

    impl<'c> AsPtr<'c> for CHistories<'_, '_, 'c> {}

    impl<'a, 'b, 'c> AsPtr<'c, CHistories<'a, 'b, 'c>> for TestHistories<'a, 'b, 'c> {
        fn as_ptr(&self) -> *const CHistories<'a, 'b, 'c> {
            self.histories.as_ptr()
        }

        fn as_mut_ptr(&mut self) -> *mut CHistories<'a, 'b, 'c> {
            self.histories.as_mut_ptr()
        }
    }

    impl Default for TestHistories<'_, '_, '_> {
        fn default() -> Self {
            let len = 6;
            let ids = Pin::new(
                (0..len)
                    .map(|idx| CString::new(idx.to_string()).unwrap())
                    .collect::<Vec<_>>(),
            );
            let relevances = repeat(CRelevance::Low)
                .take(len / 2)
                .chain(repeat(CRelevance::High).take(len - len / 2));
            let feedbacks = repeat(CFeedback::Irrelevant)
                .take(len / 2)
                .chain(repeat(CFeedback::Relevant).take(len - len / 2));

            let history = izip!(ids.as_ref().get_ref(), relevances, feedbacks)
                .map(|(id, relevance, feedback)| CHistory {
                    id: unsafe { FfiStr::from_raw(id.as_ptr()) },
                    relevance,
                    feedback,
                    _variance: PhantomData,
                })
                .collect::<Vec<_>>();
            let histories = CHistories {
                data: history.as_ptr(),
                len: len as u32,
                _variance: PhantomData,
            };

            Self {
                len,
                ids,
                history,
                histories,
                _variance: PhantomData,
            }
        }
    }

    #[allow(dead_code)]
    pub struct TestDocuments<'a, 'b, 'c, 'd> {
        pub len: usize,
        ids: Pin<Vec<CString>>,
        snippets: Pin<Vec<CString>>,
        document: Vec<CDocument<'a, 'b, 'c>>,
        documents: CDocuments<'a, 'b, 'c, 'd>,
        _variance: PhantomData<&'d Pin<Vec<CString>>>,
    }

    impl<'d> AsPtr<'d> for CDocuments<'_, '_, '_, 'd> {}

    impl<'a, 'b, 'c, 'd> AsPtr<'d, CDocuments<'a, 'b, 'c, 'd>> for TestDocuments<'a, 'b, 'c, 'd> {
        fn as_ptr(&self) -> *const CDocuments<'a, 'b, 'c, 'd> {
            self.documents.as_ptr()
        }

        fn as_mut_ptr(&mut self) -> *mut CDocuments<'a, 'b, 'c, 'd> {
            self.documents.as_mut_ptr()
        }
    }

    impl Default for TestDocuments<'_, '_, '_, '_> {
        fn default() -> Self {
            let len = 10;
            let ids = Pin::new(
                (0..len)
                    .map(|idx| CString::new(idx.to_string()).unwrap())
                    .collect::<Vec<_>>(),
            );
            let snippets = Pin::new(
                (0..len)
                    .map(|idx| CString::new(format!("snippet {}", idx)).unwrap())
                    .collect::<Vec<_>>(),
            );
            let ranks = 0..len as u32;

            let document = izip!(ids.as_ref().get_ref(), snippets.as_ref().get_ref(), ranks)
                .map(|(id, snippet, rank)| CDocument {
                    id: unsafe { FfiStr::from_raw(id.as_ptr()) },
                    snippet: unsafe { FfiStr::from_raw(snippet.as_ptr()) },
                    rank,
                    _variance: PhantomData,
                })
                .collect::<Vec<_>>();
            let documents = CDocuments {
                data: document.as_ptr(),
                len: len as u32,
                _variance: PhantomData,
            };

            Self {
                len,
                ids,
                snippets,
                document,
                documents,
                _variance: PhantomData,
            }
        }
    }

    #[test]
    fn test_histories_to_vec() {
        let hists = TestHistories::default();
        let histories = unsafe { hists.histories.to_histories() }.unwrap();
        assert_eq!(histories.len(), hists.len);
        for (dh, ch) in izip!(histories, &hists.history) {
            assert_eq!(dh.id.0, ch.id.as_str());
            assert_eq!(dh.relevance, ch.relevance.into());
            assert_eq!(dh.user_feedback, ch.feedback.into());
        }
    }

    #[test]
    fn test_histories_empty_null() {
        let mut hists = TestHistories::default();
        hists.histories.data = null();
        assert!(unsafe { hists.histories.to_histories() }
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_histories_empty_zero() {
        let mut hists = TestHistories::default();
        hists.histories.len = 0;
        assert!(unsafe { hists.histories.to_histories() }
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_history_id_null() {
        let mut hists = TestHistories::default();
        hists.history[0].id = unsafe { FfiStr::from_raw(null()) };
        let error = unsafe { hists.histories.to_histories() }.unwrap_err();
        assert_eq!(error.get_code(), CError::HistoryIdPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: A document history id is not a valid C-string pointer",
        );
    }

    #[test]
    fn test_documents_to_vec() {
        let docs = TestDocuments::default();
        let documents = unsafe { docs.documents.to_documents() }.unwrap();
        assert_eq!(documents.len(), docs.len);
        for (d, cd) in izip!(documents, &docs.document) {
            assert_eq!(d.id.0, cd.id.as_str());
            assert_eq!(d.snippet, cd.snippet.as_str());
            assert_eq!(d.rank, cd.rank as usize);
        }
    }

    #[test]
    fn test_documents_empty_null() {
        let mut docs = TestDocuments::default();
        docs.documents.data = null();
        assert!(unsafe { docs.documents.to_documents() }.unwrap().is_empty());
    }

    #[test]
    fn test_documents_empty_zero() {
        let mut docs = TestDocuments::default();
        docs.documents.len = 0;
        assert!(unsafe { docs.documents.to_documents() }.unwrap().is_empty());
    }

    #[test]
    fn test_document_id_null() {
        let mut docs = TestDocuments::default();
        docs.document[0].id = unsafe { FfiStr::from_raw(null()) };
        let error = unsafe { docs.documents.to_documents() }.unwrap_err();
        assert_eq!(error.get_code(), CError::DocumentIdPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: A document id is not a valid C-string pointer",
        );
    }

    #[test]
    fn test_document_snippet_null() {
        let mut docs = TestDocuments::default();
        docs.document[0].snippet = unsafe { FfiStr::from_raw(null()) };
        let error = unsafe { docs.documents.to_documents() }.unwrap_err();
        assert_eq!(error.get_code(), CError::DocumentSnippetPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
        );
    }
}
