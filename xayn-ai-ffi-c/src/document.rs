use ffi_support::{ErrorCode, ExternError, FfiStr};
use xayn_ai::{Document, DocumentHistory, Relevance, UserFeedback};

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

/// Collects the document history from raw.
pub fn hist_to_vec<'a>(hist: &'a [CHistory<'a>]) -> Result<Vec<DocumentHistory>, ExternError> {
    hist.iter()
        .map(|history| {
            let id = history
                .id
                .as_opt_str()
                .map(Into::into)
                .ok_or_else(|| ExternError::new_error(
                    ErrorCode::new(CXaynAiError::HistoryIdPointer as i32),
                    "Failed to rerank the documents: A document history id is not a valid C-string pointer",
                ))?;
            let relevance = history.relevance.into();
            let user_feedback = history.feedback.into();

            Ok(DocumentHistory {id, relevance, user_feedback })
        })
        .collect()
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

/// Collects the documents from raw.
pub fn docs_to_vec<'a>(docs: &'a [CDocument<'a>]) -> Result<Vec<Document>, ExternError> {
    docs.iter()
        .map(|document| {
            let id = document
                .id
                .as_opt_str()
                .map(Into::into)
                .ok_or_else(|| ExternError::new_error(
                    ErrorCode::new(CXaynAiError::DocumentIdPointer as i32),
                    "Failed to rerank the documents: A document id is not a valid C-string pointer",
                ))?;
            let snippet = document
                .snippet
                .as_opt_str()
                .map(Into::into)
                .ok_or_else(|| ExternError::new_error(
                    ErrorCode::new(CXaynAiError::DocumentSnippetPointer as i32),
                    "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
                ))?;
            let rank = document.rank as usize;

            Ok(Document { id, snippet, rank })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ptr::null;

    use super::*;
    use crate::utils::tests::{drop_values, setup_pointers, setup_values};

    #[test]
    fn test_history_to_vec() {
        let (vocab, model, hist, hist_size, docs, _, mut error) = setup_values();
        let (_, _, c_hist, _, _) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let history = hist_to_vec(c_hist.as_slice()).unwrap();
        assert_eq!(history.len(), hist_size as usize);
        for (dh, ch) in history.into_iter().zip(c_hist) {
            assert_eq!(dh.id.0, ch.id.as_str());
            assert_eq!(dh.relevance, ch.relevance.into());
            assert_eq!(dh.user_feedback, ch.feedback.into());
        }

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_history_id_null() {
        let (vocab, model, hist, _, docs, _, mut error) = setup_values();
        let (_, _, mut c_invalid, _, _) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        c_invalid[0].id = unsafe { FfiStr::from_raw(null()) };
        let error = hist_to_vec(c_invalid.as_slice()).unwrap_err();
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
        let (_, _, _, c_docs, _) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let documents = docs_to_vec(c_docs.as_slice()).unwrap();
        assert_eq!(documents.len(), docs_size as usize);
        for (d, cd) in documents.into_iter().zip(c_docs) {
            assert_eq!(d.id.0, cd.id.as_str());
            assert_eq!(d.snippet, cd.snippet.as_str());
            assert_eq!(d.rank, cd.rank as usize);
        }

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_document_id_null() {
        let (vocab, model, hist, _, docs, _, mut error) = setup_values();
        let (_, _, _, mut c_invalid, _) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        c_invalid[0].id = unsafe { FfiStr::from_raw(null()) };
        let error = docs_to_vec(c_invalid.as_slice()).unwrap_err();
        assert_eq!(error.get_code(), CXaynAiError::DocumentIdPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: A document id is not a valid C-string pointer",
        );

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_document_snippet_null() {
        let (vocab, model, hist, _, docs, _, mut error) = setup_values();
        let (_, _, _, mut c_invalid, _) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        c_invalid[0].snippet = unsafe { FfiStr::from_raw(null()) };
        let error = docs_to_vec(c_invalid.as_slice()).unwrap_err();
        assert_eq!(error.get_code(), CXaynAiError::DocumentSnippetPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
        );

        drop_values(vocab, model, hist, docs, error);
    }
}
