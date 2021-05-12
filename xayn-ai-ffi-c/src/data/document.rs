use std::{convert::TryInto, slice};

use xayn_ai::Document;

use crate::{
    result::error::{CCode, Error},
    utils::as_str,
};

/// A raw document.
#[repr(C)]
pub struct CDocument<'a> {
    /// The raw pointer to the document id.
    pub id: Option<&'a u8>,
    /// The raw pointer to the document snippet.
    pub snippet: Option<&'a u8>,
    /// The rank of the document.
    pub rank: u32,
}

/// A raw slice of documents.
#[repr(C)]
pub struct CDocuments<'a> {
    /// The raw pointer to the documents.
    pub data: Option<&'a CDocument<'a>>,
    /// The number of documents.
    pub len: u32,
}

impl<'a> CDocuments<'a> {
    /// Collects the documents from raw.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A non-null `data` doesn't point to an aligned, contiguous area of memory with at least
    /// `len` many [`CDocument`]s.
    /// - A `len` is too large to address the memory of a non-null [`CDocument`] array.
    /// - A non-null `id` or `snippet` doesn't point to an aligned, contiguous area of memory with a
    /// terminating null byte.
    pub unsafe fn to_documents(&self) -> Result<Vec<Document>, Error> {
        match (self.data, self.len) {
            (None, _) | (_, 0) => Ok(Vec::new()),
            (Some(data), len) => unsafe { slice::from_raw_parts::<'a>(data, len as usize) }
                .iter()
                .map(|document| {
                    let id = unsafe {
                        as_str(
                            document.id,
                            CCode::DocumentIdPointer,
                            "Failed to rerank the documents",
                        )
                    }
                    .and_then(|s| {
                        s.try_into().map_err(|e| {
                            CCode::DocumentIdPointer
                                .with_context(format!("Invalid uuid string: {}", e))
                        })
                    })?;
                    let snippet = unsafe {
                        as_str(
                            document.snippet,
                            CCode::DocumentSnippetPointer,
                            "Failed to rerank the documents",
                        )
                    }?
                    .into();
                    let rank = document.rank as usize;

                    Ok(Document { id, rank, snippet })
                })
                .collect(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{ffi::CString, pin::Pin};

    use itertools::izip;

    use xayn_ai::DocumentId;

    use super::*;
    use crate::utils::tests::as_str_unchecked;

    pub struct TestDocuments<'a> {
        _ids: Pin<Vec<CString>>,
        _snippets: Pin<Vec<CString>>,
        document: Pin<Vec<CDocument<'a>>>,
        documents: CDocuments<'a>,
    }

    impl Drop for TestDocuments<'_> {
        fn drop(&mut self) {}
    }

    impl Default for TestDocuments<'_> {
        fn default() -> Self {
            let len = 10;
            let _ids = Pin::new(
                (0..len)
                    .map(|idx| CString::new(DocumentId::from_u128(idx).to_string()).unwrap())
                    .collect::<Vec<_>>(),
            );
            let _snippets = Pin::new(
                (0..len)
                    .map(|idx| CString::new(format!("snippet {}", idx)).unwrap())
                    .collect::<Vec<_>>(),
            );
            let ranks = 0..len as u32;

            let document = Pin::new(
                izip!(_ids.as_ref().get_ref(), _snippets.as_ref().get_ref(), ranks)
                    .map(|(id, snippet, rank)| CDocument {
                        id: unsafe { id.as_ptr().cast::<u8>().as_ref() },
                        snippet: unsafe { snippet.as_ptr().cast::<u8>().as_ref() },
                        rank,
                    })
                    .collect::<Vec<_>>(),
            );
            let documents = CDocuments {
                data: unsafe { document.as_ptr().as_ref() },
                len: document.len() as u32,
            };

            Self {
                _ids,
                _snippets,
                document,
                documents,
            }
        }
    }

    impl<'a> TestDocuments<'a> {
        pub fn as_ptr(&self) -> Option<&CDocuments<'a>> {
            Some(&self.documents)
        }

        fn len(&self) -> usize {
            self.document.len()
        }
    }

    #[test]
    fn test_documents_to_vec() {
        let docs = TestDocuments::default();
        let documents = unsafe { docs.documents.to_documents() }.unwrap();
        assert_eq!(documents.len(), docs.len());
        for (d, cd) in izip!(documents, docs.document.as_ref().get_ref()) {
            assert_eq!(d.id.0.to_string(), as_str_unchecked(cd.id));
            assert_eq!(d.snippet, as_str_unchecked(cd.snippet));
            assert_eq!(d.rank, cd.rank as usize);
        }
    }

    #[test]
    fn test_documents_empty_null() {
        let mut docs = TestDocuments::default();
        docs.documents.data = None;
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
        docs.document[0].id = None;

        let error = unsafe { docs.documents.to_documents() }.unwrap_err();
        assert_eq!(error.code, CCode::DocumentIdPointer);
        assert_eq!(
            error.message,
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::DocumentIdPointer,
            ),
        );
    }

    #[test]
    fn test_document_snippet_null() {
        let mut docs = TestDocuments::default();
        docs.document[0].snippet = None;

        let error = unsafe { docs.documents.to_documents() }.unwrap_err();
        assert_eq!(error.code, CCode::DocumentSnippetPointer);
        assert_eq!(
            error.message,
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::DocumentSnippetPointer,
            ),
        );
    }
}
