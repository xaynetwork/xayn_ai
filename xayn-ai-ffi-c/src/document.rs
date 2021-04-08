use std::{
    collections::HashMap,
    panic::catch_unwind,
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use ffi_support::{ExternError, FfiStr, IntoFfi};
use xayn_ai::{Document, DocumentHistory, DocumentsRank, Relevance, UserFeedback};

use crate::error::CXaynAiError;

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
pub struct CHistory<'a> {
    /// The raw pointer to the document id.
    pub id: FfiStr<'a>,
    /// The relevance level of the document.
    pub relevance: CRelevance,
    /// The user feedback level of the document.
    pub feedback: CFeedback,
}

impl CHistory<'_> {
    /// Collects the document history from raw.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A history reference doesn't point to an aligned, contiguous area of memory with at least
    /// history size many [`CHistory`]s.
    /// - A history size is too large to address the memory of a history slice.
    /// - A non-null id doesn't point to an aligned, contiguous area of memory with a terminating
    /// null byte.
    pub unsafe fn to_history(&self, size: u32) -> Result<Vec<DocumentHistory>, ExternError> {
        if size == 0 {
            return Ok(Vec::new());
        }

        unsafe { from_raw_parts(self, size as usize) }
            .iter()
            .map(|history| {
                let id = history
                    .id
                    .as_opt_str()
                    .map(Into::into)
                    .ok_or_else(|| {
                        CXaynAiError::HistoryIdPointer.with_context(
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
pub struct CDocument<'a> {
    /// The raw pointer to the document id.
    pub id: FfiStr<'a>,
    /// The raw pointer to the document snippet.
    pub snippet: FfiStr<'a>,
    /// The rank of the document.
    pub rank: u32,
}

impl CDocument<'_> {
    /// Collects the documents from raw.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A documents reference doesn't point to an aligned, contiguous area of memory with at least
    /// documents size many [`CDocument`]s.
    /// - A documents size is too large to address the memory of a documents slice.
    /// - A non-null id or snippet doesn't point to an aligned, contiguous area of memory with a
    /// terminating null byte.
    pub unsafe fn to_documents(&self, size: u32) -> Result<Vec<Document>, ExternError> {
        if size == 0 {
            return Ok(Vec::new());
        }

        unsafe { from_raw_parts(self, size as usize) }
            .iter()
            .map(|document| {
                let id = document
                    .id
                    .as_opt_str()
                    .map(Into::into)
                    .ok_or_else(|| {
                        CXaynAiError::DocumentIdPointer.with_context(
                            "Failed to rerank the documents: A document id is not a valid C-string pointer",
                        )
                    })?;
                let snippet = document
                    .snippet
                    .as_opt_str()
                    .map(Into::into)
                    .ok_or_else(|| {
                        CXaynAiError::DocumentSnippetPointer.with_context(
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
                CXaynAiError::Internal.with_context(
                    "Failed to rerank the documents: The document ids are inconsistent",
                )
            })
    }

    unsafe fn drop(ranks: *mut u32, ranks_size: u32) {
        if !ranks.is_null() && ranks_size > 0 {
            unsafe { Box::from_raw(from_raw_parts_mut(ranks, ranks_size as usize)) };
        }
    }
}

/// Frees the memory of the ranks array.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null ranks doesn't point to memory allocated by [`xaynai_rerank()`].
/// - A non-zero ranks size is different from the documents size used in [`xaynai_rerank()`].
/// - A non-null ranks is freed more than once.
/// - A non-null ranks is accessed after being freed.
///
/// [`xaynai_rerank()`]: crate::ai::xaynai_rerank
#[no_mangle]
pub unsafe extern "C" fn ranks_drop(ranks: *mut u32, ranks_size: u32) {
    let _ = catch_unwind(|| unsafe { CRanks::drop(ranks, ranks_size) });
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
    let _ = catch_unwind(|| unsafe { CBytes::drop(buffer) });
}

#[cfg(test)]
mod tests {
    use std::ptr::null;

    use super::*;
    use crate::utils::tests::{drop_values, setup_pointers, setup_values};

    #[test]
    fn test_history_to_vec() {
        let (vocab, model, hist, hist_size, docs, _, mut error) = setup_values();
        let (_, _, _, c_hist, _, _) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let history = unsafe { c_hist[0].to_history(hist_size) }.unwrap();
        assert_eq!(history.len(), hist_size as usize);
        for (dh, ch) in history.into_iter().zip(c_hist) {
            assert_eq!(dh.id.0, ch.id.as_str());
            assert_eq!(dh.relevance, ch.relevance.into());
            assert_eq!(dh.user_feedback, ch.feedback.into());
        }

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_history_empty() {
        let history = unsafe { (&*null::<CHistory>()).to_history(0) }.unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn test_history_id_null() {
        let (vocab, model, hist, hist_size, docs, _, mut error) = setup_values();
        let (_, _, _, mut c_invalid, _, _) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        c_invalid[0].id = unsafe { FfiStr::from_raw(null()) };
        let error = unsafe { c_invalid[0].to_history(hist_size) }.unwrap_err();
        assert_eq!(error.get_code(), CXaynAiError::HistoryIdPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: A document history id is not a valid C-string pointer",
        );

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_documents_to_vec() {
        let (vocab, model, hist, _, docs, docs_size, mut error) = setup_values();
        let (_, _, _, _, c_docs, _) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let documents = unsafe { c_docs[0].to_documents(docs_size) }.unwrap();
        assert_eq!(documents.len(), docs_size as usize);
        for (d, cd) in documents.into_iter().zip(c_docs) {
            assert_eq!(d.id.0, cd.id.as_str());
            assert_eq!(d.snippet, cd.snippet.as_str());
            assert_eq!(d.rank, cd.rank as usize);
        }

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_documents_empty() {
        let documents = unsafe { (&*null::<CDocument>()).to_documents(0) }.unwrap();
        assert!(documents.is_empty());
    }

    #[test]
    fn test_document_id_null() {
        let (vocab, model, hist, _, docs, docs_size, mut error) = setup_values();
        let (_, _, _, _, mut c_invalid, _) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        c_invalid[0].id = unsafe { FfiStr::from_raw(null()) };
        let error = unsafe { c_invalid[0].to_documents(docs_size) }.unwrap_err();
        assert_eq!(error.get_code(), CXaynAiError::DocumentIdPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: A document id is not a valid C-string pointer",
        );

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_document_snippet_null() {
        let (vocab, model, hist, _, docs, docs_size, mut error) = setup_values();
        let (_, _, _, _, mut c_invalid, _) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        c_invalid[0].snippet = unsafe { FfiStr::from_raw(null()) };
        let error = unsafe { c_invalid[0].to_documents(docs_size) }.unwrap_err();
        assert_eq!(error.get_code(), CXaynAiError::DocumentSnippetPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
        );

        drop_values(vocab, model, hist, docs, error);
    }
}
