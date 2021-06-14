use std::{convert::TryInto, slice};

use xayn_ai::Document;
use xayn_ai_ffi::{CCode, Error};

use crate::utils::as_str;

/// A raw document.
#[repr(C)]
pub struct CDocument<'a> {
    /// The raw pointer to the document id.
    pub id: Option<&'a u8>,
    /// The raw pointer to the document title.
    pub title: Option<&'a u8>,
    /// The raw pointer to the document snippet.
    pub snippet: Option<&'a u8>,
    /// The rank of the document.
    pub rank: u32,
    /// The raw pointer to the document session id.
    pub session: Option<&'a u8>,
    /// Query count within session
    pub query_count: u32,
    /// The raw pointer to the document query id.
    pub query_id: Option<&'a u8>,
    /// The raw pointer to the document query words.
    pub query_words: Option<&'a u8>,
    /// The raw pointer to the document URL.
    pub url: Option<&'a u8>,
    /// The raw pointer to the document domain.
    pub domain: Option<&'a u8>,
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
    /// - A non-null `id` or `title` doesn't point to an aligned, contiguous area of memory with a
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
                    let title = unsafe {
                        as_str(
                            document.title,
                            CCode::DocumentTitlePointer,
                            "Failed to rerank the documents",
                        )
                    }?
                    .into();
                    let snippet = unsafe {
                        as_str(
                            document.snippet,
                            CCode::DocumentSnippetPointer,
                            "Failed to rerank the documents",
                        )
                    }?
                    .into();
                    let rank = document.rank as usize;
                    let session = unsafe {
                        as_str(
                            document.session,
                            CCode::DocumentSessionPointer,
                            "Failed to rerank the documents",
                        )
                    }
                    .and_then(|s| {
                        s.try_into().map_err(|e| {
                            CCode::DocumentSessionPointer
                                .with_context(format!("Invalid uuid string: {}", e))
                        })
                    })?;
                    let query_count = document.query_count as usize;
                    let query_id = unsafe {
                        as_str(
                            document.query_id,
                            CCode::DocumentQueryIdPointer,
                            "Failed to rerank the documents",
                        )
                    }
                    .and_then(|s| {
                        s.try_into().map_err(|e| {
                            CCode::DocumentQueryIdPointer
                                .with_context(format!("Invalid uuid string: {}", e))
                        })
                    })?;
                    let query_words = unsafe {
                        as_str(
                            document.query_words,
                            CCode::DocumentQueryWordsPointer,
                            "Failed to rerank the documents",
                        )
                    }?
                    .into();
                    let url = unsafe {
                        as_str(
                            document.url,
                            CCode::DocumentUrlPointer,
                            "Failed to rerank the documents",
                        )
                    }?
                    .into();
                    let domain = unsafe {
                        as_str(
                            document.domain,
                            CCode::DocumentDomainPointer,
                            "Failed to rerank the documents",
                        )
                    }?
                    .into();

                    Ok(Document {
                        id,
                        rank,
                        title,
                        snippet,
                        session,
                        query_count,
                        query_id,
                        query_words,
                        url,
                        domain,
                    })
                })
                .collect(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{ffi::CString, iter, pin::Pin};

    use itertools::izip;

    use xayn_ai::{DocumentId, QueryId, SessionId};

    use super::*;
    use crate::utils::tests::as_str_unchecked;

    //FIXME[philipp] code around this is unsound, has user-after free and leaks memory
    // 1. This is a self referential struct, but there are soundness issues around self
    //    referential structs. Currently the only way to work around this is to guarantee
    //    the self referential sturct is `!Unpin`. We still need to track https://github.com/rust-lang/rust/issues/63818.
    // 2. The drop implementation looks suss. At lest document it more.
    pub struct TestDocuments<'a> {
        _ids: Pin<Vec<CString>>,
        _titles: Pin<Vec<CString>>,
        _snippets: Pin<Vec<CString>>,
        _sessions: Pin<Vec<CString>>,
        _query_ids: Pin<Vec<CString>>,
        _query_words: Pin<Vec<CString>>,
        _urls: Pin<Vec<CString>>,
        _domains: Pin<Vec<CString>>,
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
            let _titles = Pin::new(
                (0..len)
                    .map(|idx| CString::new(format!("title {}", idx)).unwrap())
                    .collect::<Vec<_>>(),
            );
            let _snippets = Pin::new(
                (0..len)
                    .map(|idx| CString::new(format!("snippet {}", idx)).unwrap())
                    .collect::<Vec<_>>(),
            );
            let ranks = 0..len as u32;
            let _sessions = Pin::new(
                (0..len)
                    .map(|idx| CString::new(SessionId::from_u128(idx).to_string()).unwrap())
                    .collect::<Vec<_>>(),
            );
            let query_counts = iter::repeat(1).take(len as usize);
            let _query_ids = Pin::new(
                (0..len)
                    .map(|idx| CString::new(QueryId::from_u128(idx).to_string()).unwrap())
                    .collect::<Vec<_>>(),
            );
            let _query_words = Pin::new(
                (0..len)
                    .map(|idx| CString::new(format!("query {}", idx)).unwrap())
                    .collect::<Vec<_>>(),
            );
            let _urls = Pin::new(
                (0..len)
                    .map(|idx| CString::new(format!("url-{}", idx)).unwrap())
                    .collect::<Vec<_>>(),
            );
            let _domains = Pin::new(
                (0..len)
                    .map(|idx| CString::new(format!("domain-{}", idx)).unwrap())
                    .collect::<Vec<_>>(),
            );

            let document = Pin::new(
                izip!(
                    _ids.as_ref().get_ref(),
                    _titles.as_ref().get_ref(),
                    _snippets.as_ref().get_ref(),
                    ranks,
                    _sessions.as_ref().get_ref(),
                    query_counts,
                    _query_ids.as_ref().get_ref(),
                    _query_words.as_ref().get_ref(),
                    _urls.as_ref().get_ref(),
                    _domains.as_ref().get_ref()
                )
                .map(|cdoc| CDocument {
                    id: unsafe { cdoc.0.as_ptr().cast::<u8>().as_ref() },
                    title: unsafe { cdoc.1.as_ptr().cast::<u8>().as_ref() },
                    snippet: unsafe { cdoc.2.as_ptr().cast::<u8>().as_ref() },
                    rank: cdoc.3,
                    session: unsafe { cdoc.4.as_ptr().cast::<u8>().as_ref() },
                    query_count: cdoc.5,
                    query_id: unsafe { cdoc.6.as_ptr().cast::<u8>().as_ref() },
                    query_words: unsafe { cdoc.7.as_ptr().cast::<u8>().as_ref() },
                    url: unsafe { cdoc.8.as_ptr().cast::<u8>().as_ref() },
                    domain: unsafe { cdoc.9.as_ptr().cast::<u8>().as_ref() },
                })
                .collect::<Vec<_>>(),
            );

            let documents = CDocuments {
                data: unsafe { document.as_ptr().as_ref() },
                len: document.len() as u32,
            };

            Self {
                _ids,
                _titles,
                _snippets,
                _sessions,
                _query_ids,
                _query_words,
                _urls,
                _domains,
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
            assert_eq!(d.id.0.to_string(), unsafe { as_str_unchecked(cd.id) });
            assert_eq!(d.title, unsafe { as_str_unchecked(cd.title) });
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
        assert_eq!(error.code(), CCode::DocumentIdPointer);
        assert_eq!(
            error.message(),
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::DocumentIdPointer,
            ),
        );
    }

    #[test]
    fn test_document_title_null() {
        let mut docs = TestDocuments::default();
        docs.document[0].title = None;

        let error = unsafe { docs.documents.to_documents() }.unwrap_err();
        assert_eq!(error.code(), CCode::DocumentTitlePointer);
        assert_eq!(
            error.message(),
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::DocumentTitlePointer,
            ),
        );
    }
}
