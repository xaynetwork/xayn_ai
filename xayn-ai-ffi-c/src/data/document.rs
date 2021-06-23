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
    /// - A non-null pointer of a "text" field in in any [`CDocument`] does not point to an aligned,
    ///   contiguous area of memory with a terminating null bytes (`id` is a "text" field).
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
    use std::{ffi::CString, iter, marker::PhantomPinned, pin::Pin};

    use itertools::izip;
    use xayn_ai::{DocumentId, QueryId, SessionId};

    use super::*;
    use crate::utils::tests::as_str_unchecked;

    pub struct TestDocuments<'a> {
        ids: Vec<CString>,
        titles: Vec<CString>,
        snippets: Vec<CString>,
        sessions: Vec<CString>,
        query_ids: Vec<CString>,
        query_words: Vec<CString>,
        urls: Vec<CString>,
        domains: Vec<CString>,
        document: Vec<CDocument<'a>>,
        documents: CDocuments<'a>,
        _pinned: PhantomPinned,
    }

    impl<'a> TestDocuments<'a> {
        fn uninitialized() -> Pin<Box<Self>> {
            let len = 10;
            let ids = (0..len)
                .map(|idx| CString::new(DocumentId::from_u128(idx).to_string()).unwrap())
                .collect::<Vec<_>>();
            let titles = (0..len)
                .map(|idx| CString::new(format!("title {}", idx)).unwrap())
                .collect::<Vec<_>>();
            let snippets = (0..len)
                .map(|idx| CString::new(format!("snippet {}", idx)).unwrap())
                .collect::<Vec<_>>();
            let sessions = (0..len)
                .map(|idx| CString::new(SessionId::from_u128(idx).to_string()).unwrap())
                .collect::<Vec<_>>();
            let query_ids = (0..len)
                .map(|idx| CString::new(QueryId::from_u128(idx).to_string()).unwrap())
                .collect::<Vec<_>>();
            let query_words = (0..len)
                .map(|idx| CString::new(format!("query {}", idx)).unwrap())
                .collect::<Vec<_>>();
            let urls = (0..len)
                .map(|idx| CString::new(format!("url-{}", idx)).unwrap())
                .collect::<Vec<_>>();
            let domains = (0..len)
                .map(|idx| CString::new(format!("domain-{}", idx)).unwrap())
                .collect::<Vec<_>>();

            Box::pin(Self {
                ids,
                titles,
                snippets,
                sessions,
                query_ids,
                query_words,
                urls,
                domains,
                document: Vec::new(),
                documents: CDocuments { data: None, len: 0 },
                _pinned: PhantomPinned,
            })
        }

        fn initialize_document(mut self: Pin<Box<Self>>) -> Pin<Box<Self>> {
            let len = self.len();
            let ranks = 0..len as u32;
            let query_counts = iter::repeat(1).take(len);

            let document = izip!(
                self.ids.iter(),
                self.titles.iter(),
                self.snippets.iter(),
                ranks,
                self.sessions.iter(),
                query_counts,
                self.query_ids.iter(),
                self.query_words.iter(),
                self.urls.iter(),
                self.domains.iter(),
            )
            .map(
                |(
                    id,
                    title,
                    snippet,
                    rank,
                    session,
                    query_count,
                    query_id,
                    query_words,
                    url,
                    domain,
                )| {
                    CDocument {
                        id: unsafe { id.as_ptr().cast::<u8>().as_ref() },
                        title: unsafe { title.as_ptr().cast::<u8>().as_ref() },
                        snippet: unsafe { snippet.as_ptr().cast::<u8>().as_ref() },
                        rank,
                        session: unsafe { session.as_ptr().cast::<u8>().as_ref() },
                        query_count,
                        query_id: unsafe { query_id.as_ptr().cast::<u8>().as_ref() },
                        query_words: unsafe { query_words.as_ptr().cast::<u8>().as_ref() },
                        url: unsafe { url.as_ptr().cast::<u8>().as_ref() },
                        domain: unsafe { domain.as_ptr().cast::<u8>().as_ref() },
                    }
                },
            )
            .collect::<Vec<_>>();
            unsafe { self.as_mut().get_unchecked_mut() }.document = document;

            self
        }

        fn initialize_documents(mut self: Pin<Box<Self>>) -> Pin<Box<Self>> {
            let data = unsafe { self.document.as_ptr().as_ref() };
            let len = self.len() as u32;
            unsafe { self.as_mut().get_unchecked_mut() }.documents = CDocuments { data, len };

            self
        }

        pub fn initialized() -> Pin<Box<Self>> {
            Self::uninitialized()
                .initialize_document()
                .initialize_documents()
        }

        #[allow(clippy::wrong_self_convention)] // false positive
        pub fn as_ptr(self: &'a Pin<Box<Self>>) -> Option<&'a CDocuments<'a>> {
            Some(&self.documents)
        }

        fn len(self: &Pin<Box<Self>>) -> usize {
            self.ids.len()
        }
    }

    #[test]
    fn test_documents_to_vec() {
        let docs = TestDocuments::initialized();
        let documents = unsafe { docs.documents.to_documents() }.unwrap();
        assert_eq!(documents.len(), docs.len());
        for (d, cd) in izip!(documents, docs.document.iter()) {
            assert_eq!(d.id.0.to_string(), unsafe { as_str_unchecked(cd.id) });
            assert_eq!(d.title, unsafe { as_str_unchecked(cd.title) });
            assert_eq!(d.rank, cd.rank as usize);
        }
    }

    #[test]
    fn test_documents_empty_null() {
        let mut docs = TestDocuments::initialized();
        unsafe { docs.as_mut().get_unchecked_mut() }.documents.data = None;
        assert!(unsafe { docs.documents.to_documents() }.unwrap().is_empty());
    }

    #[test]
    fn test_documents_empty_zero() {
        let mut docs = TestDocuments::initialized();
        unsafe { docs.as_mut().get_unchecked_mut() }.documents.len = 0;
        assert!(unsafe { docs.documents.to_documents() }.unwrap().is_empty());
    }

    #[test]
    fn test_document_id_null() {
        let mut docs = TestDocuments::uninitialized().initialize_document();
        unsafe { docs.as_mut().get_unchecked_mut() }.document[0].id = None;
        let docs = docs.initialize_documents();

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
        let mut docs = TestDocuments::uninitialized().initialize_document();
        unsafe { docs.as_mut().get_unchecked_mut() }.document[0].title = None;
        let docs = docs.initialize_documents();

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
