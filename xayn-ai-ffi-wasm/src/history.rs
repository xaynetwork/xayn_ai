use serde::{Deserialize, Serialize};
use xayn_ai::{DocumentHistory, DocumentId, QueryId, SessionId};
use xayn_ai_ffi::{CDayOfWeek, CFeedback, CRelevance, CUserAction};

/// A document history.
///
/// The enum fields are serializable as integers instead of strings.
#[derive(Deserialize, Serialize)]
pub struct WHistory {
    /// Unique identifier of the document.
    id: DocumentId,
    /// Relevance level of the document.
    relevance: CRelevance,
    /// A flag that indicates whether the user liked the document.
    feedback: CFeedback,
    /// Session of the document.
    session: SessionId,
    /// Query count within session.
    query_count: usize,
    /// Query identifier of the document.
    query_id: QueryId,
    /// Query of the document.
    query_words: String,
    /// Day of week query was performed.
    day: CDayOfWeek,
    /// URL of the document.
    url: String,
    /// Domain of the document.
    domain: String,
    /// Reranked position of the document.
    rank: usize,
    /// User interaction for the document.
    user_action: CUserAction,
}

impl From<WHistory> for DocumentHistory {
    fn from(history: WHistory) -> Self {
        Self {
            id: history.id,
            relevance: history.relevance.into(),
            user_feedback: history.feedback.into(),
            session: history.session,
            query_count: history.query_count,
            query_id: history.query_id,
            query_words: history.query_words,
            day: history.day.into(),
            url: history.url,
            domain: history.domain,
            rank: history.rank,
            user_action: history.user_action.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl From<DocumentHistory> for WHistory {
        fn from(history: DocumentHistory) -> Self {
            Self {
                id: history.id,
                relevance: history.relevance.into(),
                feedback: history.user_feedback.into(),
                session: history.session,
                query_count: history.query_count,
                query_id: history.query_id,
                query_words: history.query_words,
                day: history.day.into(),
                url: history.url,
                domain: history.domain,
                rank: history.rank,
                user_action: history.user_action.into(),
            }
        }
    }
}
