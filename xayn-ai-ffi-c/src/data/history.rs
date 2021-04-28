use std::slice::from_raw_parts;

use xayn_ai::{DocumentHistory, Relevance, UserFeedback};

use crate::{
    result::error::{CCode, Error},
    utils::CStrPtr,
};

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
    pub id: CStrPtr<'a>,
    /// The relevance level of the document.
    pub relevance: CRelevance,
    /// The user feedback level of the document.
    pub feedback: CFeedback,
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
            (Some(data), len) => unsafe { from_raw_parts::<'a>(data, len as usize) }
                .iter()
                .map(|history| {
                    let id = unsafe {
                        history
                            .id
                            .as_str(CCode::HistoryIdPointer, "Failed to rerank the documents")
                    }?
                    .into();
                    let relevance = history.relevance.into();
                    let user_feedback = history.feedback.into();

                    Ok(DocumentHistory {
                        id,
                        relevance,
                        user_feedback,
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

    pub struct TestHistories<'a> {
        _ids: Pin<Vec<CString>>,
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
                    .map(|idx| CString::new(idx.to_string()).unwrap())
                    .collect::<Vec<_>>(),
            );
            let relevances = repeat(CRelevance::Low)
                .take(len / 2)
                .chain(repeat(CRelevance::High).take(len - len / 2));
            let feedbacks = repeat(CFeedback::Irrelevant)
                .take(len / 2)
                .chain(repeat(CFeedback::Relevant).take(len - len / 2));

            let history = Pin::new(
                izip!(_ids.as_ref().get_ref(), relevances, feedbacks)
                    .map(|(id, relevance, feedback)| CHistory {
                        id: id.into(),
                        relevance,
                        feedback,
                    })
                    .collect::<Vec<_>>(),
            );
            let histories = CHistories {
                data: unsafe { history.as_ptr().as_ref() },
                len: history.len() as u32,
            };

            Self {
                _ids,
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
            assert_eq!(dh.id.0, ch.id.as_str_unchecked());
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
        hists.history[0].id = CStrPtr(None);

        let error = unsafe { hists.histories.to_histories() }.unwrap_err();
        assert_eq!(error.code, CCode::HistoryIdPointer);
        assert_eq!(
            error.message,
            format!(
                "Failed to rerank the documents: The {} is null",
                CCode::HistoryIdPointer,
            ),
        );
    }
}
