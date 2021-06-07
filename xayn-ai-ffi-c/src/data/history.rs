use std::{convert::TryInto, slice};

use xayn_ai::DocumentHistory;
use xayn_ai_ffi::{CCode, CDayOfWeek, CFeedback, CRelevance, CUserAction, Error};

use crate::utils::as_str;

/// A raw document history.
#[repr(C)]
pub struct CHistory<'a> {
    /// The raw pointer to the document id.
    pub id: Option<&'a u8>,
    /// The relevance level of the document.
    pub relevance: CRelevance,
    /// The user feedback level of the document.
    pub feedback: CFeedback,
    /// The raw pointer to the session id of the document.
    pub session: Option<&'a u8>,
    /// The query count within the session.
    pub query_count: u32,
    /// The raw pointer to the query id of the document.
    pub query_id: Option<&'a u8>,
    /// The raw pointer to the query words.
    pub query_words: Option<&'a u8>,
    /// The day of the week the query was performed.
    pub day: CDayOfWeek,
    /// The raw pointer to the url of the document.
    pub url: Option<&'a u8>,
    /// The raw pointer to the domain of the document.
    pub domain: Option<&'a u8>,
    /// The rank of the document.
    pub rank: u32,
    /// The user interaction for the document.
    pub user_action: CUserAction,
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
    /// - A non-null `id` doesn't point to an aligned, contiguous area of memory with a terminating
    /// null byte.
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
                    let relevance = history.relevance.into();
                    let user_feedback = history.feedback.into();
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
                    let day = history.day.into();
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
                    let user_action = history.user_action.into();

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
    use std::{ffi::CString, iter::repeat, pin::Pin};

    use itertools::izip;

    use super::*;
    use crate::utils::tests::as_str_unchecked;
    use xayn_ai::{DocumentId, QueryId, SessionId};

    pub struct TestHistories<'a> {
        _ids: Pin<Vec<CString>>,
        _sessions: Pin<Vec<CString>>,
        _query_ids: Pin<Vec<CString>>,
        _query_words: Pin<Vec<CString>>,
        _urls: Pin<Vec<CString>>,
        _domains: Pin<Vec<CString>>,
        history: Pin<Vec<CHistory<'a>>>,
        histories: CHistories<'a>,
    }

    impl Drop for TestHistories<'_> {
        fn drop(&mut self) {}
    }

    impl Default for TestHistories<'_> {
        fn default() -> Self {
            let len = 6;
            let _ids = Pin::new(
                (0..len)
                    .map(|idx| {
                        CString::new(DocumentId::from_u128(idx as u128).to_string()).unwrap()
                    })
                    .collect::<Vec<_>>(),
            );
            let relevances = repeat(CRelevance::Low)
                .take(len / 2)
                .chain(repeat(CRelevance::High).take(len - len / 2));
            let feedbacks = repeat(CFeedback::Irrelevant)
                .take(len / 2)
                .chain(repeat(CFeedback::Relevant).take(len - len / 2));
            let _sessions = Pin::new(
                (0..len)
                    .map(|idx| CString::new(SessionId::from_u128(idx as u128).to_string()).unwrap())
                    .collect::<Vec<_>>(),
            );
            let query_counts = repeat(1).take(len);
            let _query_ids = Pin::new(
                (0..len)
                    .map(|idx| CString::new(QueryId::from_u128(idx as u128).to_string()).unwrap())
                    .collect::<Vec<_>>(),
            );
            let _query_words = Pin::new(
                (0..len)
                    .map(|idx| CString::new(format!("query {}", idx)).unwrap())
                    .collect::<Vec<_>>(),
            );
            let days = repeat(CDayOfWeek::Sun)
                .take(len / 2)
                .chain(repeat(CDayOfWeek::Mon).take(len - len / 2));
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
            let ranks = 0..len as u32;
            let user_actions = repeat(CUserAction::Miss)
                .take(len / 2)
                .chain(repeat(CUserAction::Click).take(len - len / 2));

            let history = Pin::new(
                izip!(
                    _ids.as_ref().get_ref(),
                    relevances,
                    feedbacks,
                    _sessions.as_ref().get_ref(),
                    query_counts,
                    _query_ids.as_ref().get_ref(),
                    _query_words.as_ref().get_ref(),
                    days,
                    _urls.as_ref().get_ref(),
                    _domains.as_ref().get_ref(),
                    ranks,
                    user_actions,
                )
                .map(|chist| CHistory {
                    id: unsafe { chist.0.as_ptr().cast::<u8>().as_ref() },
                    relevance: chist.1,
                    feedback: chist.2,
                    session: unsafe { chist.3.as_ptr().cast::<u8>().as_ref() },
                    query_count: chist.4,
                    query_id: unsafe { chist.5.as_ptr().cast::<u8>().as_ref() },
                    query_words: unsafe { chist.6.as_ptr().cast::<u8>().as_ref() },
                    day: chist.7,
                    url: unsafe { chist.8.as_ptr().cast::<u8>().as_ref() },
                    domain: unsafe { chist.9.as_ptr().cast::<u8>().as_ref() },
                    rank: chist.10,
                    user_action: chist.11,
                })
                .collect::<Vec<_>>(),
            );
            let histories = CHistories {
                data: unsafe { history.as_ptr().as_ref() },
                len: history.len() as u32,
            };

            Self {
                _ids,
                _sessions,
                _query_ids,
                _query_words,
                _urls,
                _domains,
                history,
                histories,
            }
        }
    }

    impl<'a> TestHistories<'a> {
        pub fn as_ptr(&self) -> Option<&CHistories<'a>> {
            Some(&self.histories)
        }

        fn len(&self) -> usize {
            self.history.len()
        }
    }

    #[test]
    fn test_histories_to_vec() {
        let hists = TestHistories::default();
        let histories = unsafe { hists.histories.to_histories() }.unwrap();
        assert_eq!(histories.len(), hists.len());
        for (dh, ch) in izip!(histories, hists.history.as_ref().get_ref()) {
            assert_eq!(dh.id.0.to_string(), unsafe { as_str_unchecked(ch.id) });
            assert_eq!(dh.relevance, ch.relevance.into());
            assert_eq!(dh.user_feedback, ch.feedback.into());
        }
    }

    #[test]
    fn test_histories_empty_null() {
        let mut hists = TestHistories::default();
        hists.histories.data = None;
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
        hists.history[0].id = None;

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
}
