use std::{marker::PhantomData, slice::from_raw_parts};

use ffi_support::{ExternError, FfiStr};
use xayn_ai::Document;

use crate::result::error::CCode;

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

/// A raw slice of documents.
#[repr(C)]
pub struct CDocuments<'a> {
    /// The raw pointer to the documents.
    pub data: *const CDocument<'a>,
    /// The number of documents.
    pub len: u32,
    // covariant in lifetime and type
    _lifetime: PhantomData<&'a [CDocument<'a>]>,
}

impl CDocuments<'_> {
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
                        CCode::DocumentIdPointer.with_context(
                            "Failed to rerank the documents: A document id is not a valid C-string pointer",
                        )
                    })?;
                let snippet = document
                    .snippet
                    .as_opt_str()
                    .map(Into::into)
                    .ok_or_else(|| {
                        CCode::DocumentSnippetPointer.with_context(
                            "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
                        )
                    })?;
                let rank = document.rank as usize;

                Ok(Document { id, snippet, rank })
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{ffi::CString, pin::Pin, ptr::null};

    use itertools::izip;

    use super::*;
    use crate::utils::tests::AsPtr;

    #[allow(dead_code)]
    pub struct TestDocuments<'a> {
        pub len: usize,
        ids: Pin<Vec<CString>>,
        snippets: Pin<Vec<CString>>,
        document: Vec<CDocument<'a>>,
        documents: CDocuments<'a>,
        _lifetime: PhantomData<&'a Pin<Vec<CString>>>,
    }

    impl AsPtr for CDocuments<'_> {}

    impl<'a> AsPtr<CDocuments<'a>> for TestDocuments<'a> {
        fn as_ptr(&self) -> *const CDocuments<'a> {
            self.documents.as_ptr()
        }

        fn as_mut_ptr(&mut self) -> *mut CDocuments<'a> {
            self.documents.as_mut_ptr()
        }
    }

    impl Default for TestDocuments<'_> {
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
                })
                .collect::<Vec<_>>();
            let documents = CDocuments {
                data: document.as_ptr(),
                len: len as u32,
                _lifetime: PhantomData,
            };

            Self {
                len,
                ids,
                snippets,
                document,
                documents,
                _lifetime: PhantomData,
            }
        }
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
        assert_eq!(error.get_code(), CCode::DocumentIdPointer);
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
        assert_eq!(error.get_code(), CCode::DocumentSnippetPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
        );
    }
}
