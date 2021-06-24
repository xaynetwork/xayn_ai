use std::{convert::TryInto, slice};

use xayn_ai::{DayOfWeek, DocumentHistory, Relevance, UserAction, UserFeedback};
use xayn_ai_ffi::{CCode, Error};

use crate::utils::as_str;

/// A raw document history.
#[repr(C)]
pub struct CHistory<'a> {
    /// The raw pointer to the document id.
    pub id: Option<&'a u8>,
    /// The relevance level of the document.
    pub relevance: Relevance,
    /// The user feedback level of the document.
    pub user_feedback: UserFeedback,
    /// The raw pointer to the session id of the document.
    pub session: Option<&'a u8>,
    /// The query count within the session.
    pub query_count: u32,
    /// The raw pointer to the query id of the document.
    pub query_id: Option<&'a u8>,
    /// The raw pointer to the query words.
    pub query_words: Option<&'a u8>,
    /// The day of the week the query was performed.
    pub day: DayOfWeek,
    /// The raw pointer to the url of the document.
    pub url: Option<&'a u8>,
    /// The raw pointer to the domain of the document.
    pub domain: Option<&'a u8>,
    /// The rank of the document.
    pub rank: u32,
    /// The user interaction for the document.
    pub user_action: UserAction,
}

/// A raw slice of document histories.
#[repr(C)]
pub struct CHistories<'a> {
    /// The raw pointer to the document histories.
    pub data: Option<&'a CHistory<'a>>,
    /// The number of document histories.
    pub len: u32,
}

impl<'a> CHistories<'a> {
    /// Collects the document histories from raw.
    ///
    /// # Safety
    /// The behavior is undefined if:
    /// - A non-null `data` doesn't point to an aligned, contiguous area of memory with at least
    /// `len` many [`CHistory`]s.
    /// - A `len` is too large to address the memory of a non-null [`CHistory`] array.
    /// - A non-null pointer of a "text" field in in any [`CHistory`] does not point to an aligned,
    ///   contiguous area of memory with a terminating null bytes (`id` is a "text" field).
    pub unsafe fn to_histories(&self) -> Result<Vec<DocumentHistory>, Error> {
        match (self.data, self.len) {
            (None, _) | (_, 0) => Ok(Vec::new()),
            (Some(data), len) => unsafe { slice::from_raw_parts::<'a>(data, len as usize) }
                .iter()
                .map(|history| {
                    let id = unsafe {
                        as_str(
                            history.id,
                            CCode::HistoryIdPointer,
                            "Failed to rerank the documents",
                        )
                    }
                    .and_then(|s| {
                        s.try_into().map_err(|e| {
                            CCode::HistoryIdPointer
                                .with_context(format!("Invalid uuid string: {}", e))
                        })
                    })?;
                    let relevance = history.relevance;
                    let user_feedback = history.user_feedback;
                    let session = unsafe {
                        as_str(
                            history.session,
                            CCode::HistorySessionPointer,
                            "Failed to rerank the documents",
                        )
                    }
                    .and_then(|s| {
                        s.try_into().map_err(|e| {
                            CCode::HistorySessionPointer
                                .with_context(format!("Invalid uuid string: {}", e))
                        })
                    })?;
                    let query_count = history.query_count as usize;
                    let query_id = unsafe {
                        as_str(
                            history.query_id,
                            CCode::HistoryQueryIdPointer,
                            "Failed to rerank the documents",
                        )
                    }
                    .and_then(|s| {
                        s.try_into().map_err(|e| {
                            CCode::HistoryQueryIdPointer
                                .with_context(format!("Invalid uuid string: {}", e))
                        })
                    })?;
                    let query_words = unsafe {
                        as_str(
                            history.query_words,
                            CCode::HistoryQueryWordsPointer,
                            "Failed to rerank the documents",
                        )
                    }?
                    .into();
                    let day = history.day;
                    let url = unsafe {
                        as_str(
                            history.url,
                            CCode::HistoryUrlPointer,
                            "Failed to rerank the documents",
                        )
                    }?
                    .into();
                    let domain = unsafe {
                        as_str(
                            history.domain,
                            CCode::HistoryDomainPointer,
                            "Failed to rerank the documents",
                        )
                    }?
                    .into();
                    let rank = history.rank as usize;
                    let user_action = history.user_action;

                    Ok(DocumentHistory {
                        id,
                        relevance,
                        user_feedback,
                        session,
                        query_count,
                        query_id,
                        query_words,
                        day,
                        url,
                        domain,
                        rank,
                        user_action,
                    })
                })
                .collect(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{ffi::CString, iter::repeat, marker::PhantomPinned, pin::Pin};

    use itertools::izip;
    use xayn_ai::{DocumentId, QueryId, SessionId};

    use super::*;
    use crate::utils::tests::as_str_unchecked;

    pub struct TestHistories<'a> {
        ids: Vec<CString>,
        sessions: Vec<CString>,
        query_ids: Vec<CString>,
        query_words: Vec<CString>,
        urls: Vec<CString>,
        domains: Vec<CString>,
        history: Vec<CHistory<'a>>,
        histories: CHistories<'a>,
        _pinned: PhantomPinned,
    }

    impl<'a> TestHistories<'a> {
        fn uninitialized() -> Pin<Box<Self>> {
            let len = 6;
            let ids = (0..len)
                .map(|idx| CString::new(DocumentId::from_u128(idx as u128).to_string()).unwrap())
                .collect::<Vec<_>>();
            let sessions = (0..len)
                .map(|idx| CString::new(SessionId::from_u128(idx as u128).to_string()).unwrap())
                .collect::<Vec<_>>();
            let query_ids = (0..len)
                .map(|idx| CString::new(QueryId::from_u128(idx as u128).to_string()).unwrap())
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
                sessions,
                query_ids,
                query_words,
                urls,
                domains,
                history: Vec::new(),
                histories: CHistories { data: None, len: 0 },
                _pinned: PhantomPinned,
            })
        }

        fn initialize_history(mut self: Pin<Box<Self>>) -> Pin<Box<Self>> {
            let len = self.len();
            let relevances = repeat(Relevance::Low)
                .take(len / 2)
                .chain(repeat(Relevance::High).take(len - len / 2));
            let user_feedbacks = repeat(UserFeedback::Irrelevant)
                .take(len / 2)
                .chain(repeat(UserFeedback::Relevant).take(len - len / 2));
            let query_counts = repeat(1).take(len);
            let days = repeat(DayOfWeek::Sun)
                .take(len / 2)
                .chain(repeat(DayOfWeek::Mon).take(len - len / 2));
            let ranks = 0..len as u32;
            let user_actions = repeat(UserAction::Miss)
                .take(len / 2)
                .chain(repeat(UserAction::Click).take(len - len / 2));

            let history = izip!(
                self.ids.iter(),
                relevances,
                user_feedbacks,
                self.sessions.iter(),
                query_counts,
                self.query_ids.iter(),
                self.query_words.iter(),
                days,
                self.urls.iter(),
                self.domains.iter(),
                ranks,
                user_actions,
            )
            .map(
                |(
                    id,
                    relevance,
                    user_feedback,
                    session,
                    query_count,
                    query_id,
                    query_words,
                    day,
                    url,
                    domain,
                    rank,
                    user_action,
                )| CHistory {
                    id: unsafe { id.as_ptr().cast::<u8>().as_ref() },
                    relevance,
                    user_feedback,
                    session: unsafe { session.as_ptr().cast::<u8>().as_ref() },
                    query_count,
                    query_id: unsafe { query_id.as_ptr().cast::<u8>().as_ref() },
                    query_words: unsafe { query_words.as_ptr().cast::<u8>().as_ref() },
                    day,
                    url: unsafe { url.as_ptr().cast::<u8>().as_ref() },
                    domain: unsafe { domain.as_ptr().cast::<u8>().as_ref() },
                    rank,
                    user_action,
                },
            )
            .collect::<Vec<_>>();
            unsafe { self.as_mut().get_unchecked_mut() }.history = history;

            self
        }

        fn initialize_histories(mut self: Pin<Box<Self>>) -> Pin<Box<Self>> {
            let data = unsafe { self.history.as_ptr().as_ref() };
            let len = self.len() as u32;
            unsafe { self.as_mut().get_unchecked_mut() }.histories = CHistories { data, len };

            self
        }

        pub fn initialized() -> Pin<Box<Self>> {
            Self::uninitialized()
                .initialize_history()
                .initialize_histories()
        }

        #[allow(clippy::wrong_self_convention)] // false positive
        pub fn as_ptr(self: &'a Pin<Box<Self>>) -> Option<&'a CHistories<'a>> {
            Some(&self.histories)
        }

        fn len(self: &Pin<Box<Self>>) -> usize {
            self.ids.len()
        }
    }

    #[test]
    fn test_histories_to_vec() {
        let hists = TestHistories::initialized();
        let histories = unsafe { hists.histories.to_histories() }.unwrap();
        assert_eq!(histories.len(), hists.len());
        for (dh, ch) in izip!(histories, hists.history.iter()) {
            assert_eq!(dh.id.0.to_string(), unsafe { as_str_unchecked(ch.id) });
            assert_eq!(dh.relevance, ch.relevance);
            assert_eq!(dh.user_feedback, ch.user_feedback);
        }
    }

    #[test]
    fn test_histories_empty_null() {
        let mut hists = TestHistories::initialized();
        unsafe { hists.as_mut().get_unchecked_mut() }.histories.data = None;
        assert!(unsafe { hists.histories.to_histories() }
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_histories_empty_zero() {
        let mut hists = TestHistories::initialized();
        unsafe { hists.as_mut().get_unchecked_mut() }.histories.len = 0;
        assert!(unsafe { hists.histories.to_histories() }
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_history_id_null() {
        let mut hists = TestHistories::uninitialized().initialize_history();
        unsafe { hists.as_mut().get_unchecked_mut() }.history[0].id = None;
        let hists = hists.initialize_histories();

        let error = unsafe { hists.histories.to_histories() }.unwrap_err();
        assert_eq!(error.code(), CCode::HistoryIdPointer);
        assert_eq!(
            error.message(),
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::HistoryIdPointer,
            ),
        );
    }

    #[test]
    fn test_history_session_null() {
        let mut hists = TestHistories::uninitialized().initialize_history();
        unsafe { hists.as_mut().get_unchecked_mut() }.history[0].session = None;
        let hists = hists.initialize_histories();

        let error = unsafe { hists.histories.to_histories() }.unwrap_err();
        assert_eq!(error.code(), CCode::HistorySessionPointer);
        assert_eq!(
            error.message(),
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::HistorySessionPointer,
            ),
        );
    }

    #[test]
    fn test_history_queryid_null() {
        let mut hists = TestHistories::uninitialized().initialize_history();
        unsafe { hists.as_mut().get_unchecked_mut() }.history[0].query_id = None;
        let hists = hists.initialize_histories();

        let error = unsafe { hists.histories.to_histories() }.unwrap_err();
        assert_eq!(error.code(), CCode::HistoryQueryIdPointer);
        assert_eq!(
            error.message(),
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::HistoryQueryIdPointer,
            ),
        );
    }

    #[test]
    fn test_history_querywords_null() {
        let mut hists = TestHistories::uninitialized().initialize_history();
        unsafe { hists.as_mut().get_unchecked_mut() }.history[0].query_words = None;
        let hists = hists.initialize_histories();

        let error = unsafe { hists.histories.to_histories() }.unwrap_err();
        assert_eq!(error.code(), CCode::HistoryQueryWordsPointer);
        assert_eq!(
            error.message(),
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::HistoryQueryWordsPointer,
            ),
        );
    }

    #[test]
    fn test_history_url_null() {
        let mut hists = TestHistories::uninitialized().initialize_history();
        unsafe { hists.as_mut().get_unchecked_mut() }.history[0].url = None;
        let hists = hists.initialize_histories();

        let error = unsafe { hists.histories.to_histories() }.unwrap_err();
        assert_eq!(error.code(), CCode::HistoryUrlPointer);
        assert_eq!(
            error.message(),
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::HistoryUrlPointer,
            ),
        );
    }

    #[test]
    fn test_history_domain_null() {
        let mut hists = TestHistories::uninitialized().initialize_history();
        unsafe { hists.as_mut().get_unchecked_mut() }.history[0].domain = None;
        let hists = hists.initialize_histories();

        let error = unsafe { hists.histories.to_histories() }.unwrap_err();
        assert_eq!(error.code(), CCode::HistoryDomainPointer);
        assert_eq!(
            error.message(),
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::HistoryDomainPointer,
            ),
        );
    }
}
