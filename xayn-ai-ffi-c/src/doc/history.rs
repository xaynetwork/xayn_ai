use std::{marker::PhantomData, slice::from_raw_parts};

use ffi_support::{ExternError, FfiStr};
use xayn_ai::{DocumentHistory, Relevance, UserFeedback};

use crate::result::error::CError;

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
}
