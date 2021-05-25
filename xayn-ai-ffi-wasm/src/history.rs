use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};
use xayn_ai::{DayOfWeek, DocumentHistory, DocumentId, QueryId, SessionId, UserAction};
use xayn_ai_ffi::{CFeedback, CRelevance};

/// Day of the week.
#[repr(u8)]
#[derive(Deserialize_repr, Serialize_repr)]
#[cfg_attr(test, derive(Clone))]
pub enum WDayOfWeek {
    Mon = 0,
    Tue = 1,
    Wed = 2,
    Thu = 3,
    Fri = 4,
    Sat = 5,
    Sun = 6,
}

impl From<WDayOfWeek> for DayOfWeek {
    fn from(day: WDayOfWeek) -> Self {
        match day {
            WDayOfWeek::Mon => Self::Mon,
            WDayOfWeek::Tue => Self::Tue,
            WDayOfWeek::Wed => Self::Wed,
            WDayOfWeek::Thu => Self::Thu,
            WDayOfWeek::Fri => Self::Fri,
            WDayOfWeek::Sat => Self::Sat,
            WDayOfWeek::Sun => Self::Sun,
        }
    }
}

/// A user interaction.
#[repr(u8)]
#[derive(Deserialize_repr, Serialize_repr)]
#[cfg_attr(test, derive(Clone))]
pub enum WUserAction {
    Miss = 0,
    Skip = 1,
    Click = 2,
}

impl From<WUserAction> for UserAction {
    fn from(user_action: WUserAction) -> Self {
        match user_action {
            WUserAction::Miss => Self::Miss,
            WUserAction::Skip => Self::Skip,
            WUserAction::Click => Self::Click,
        }
    }
}

/// A document history.
///
/// The enum fields are serializable as integers instead of strings.
#[derive(Deserialize, Serialize)]
pub struct WHistory {
    id: DocumentId,
    relevance: CRelevance,
    feedback: CFeedback,
    session: SessionId,
    query_count: usize,
    query_id: QueryId,
    query_words: String,
    day: WDayOfWeek,
    url: String,
    domain: String,
    rank: usize,
    user_action: WUserAction,
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

    impl From<DayOfWeek> for WDayOfWeek {
        fn from(day: DayOfWeek) -> Self {
            match day {
                DayOfWeek::Mon => Self::Mon,
                DayOfWeek::Tue => Self::Tue,
                DayOfWeek::Wed => Self::Wed,
                DayOfWeek::Thu => Self::Thu,
                DayOfWeek::Fri => Self::Fri,
                DayOfWeek::Sat => Self::Sat,
                DayOfWeek::Sun => Self::Sun,
            }
        }
    }

    impl From<UserAction> for WUserAction {
        fn from(user_action: UserAction) -> Self {
            match user_action {
                UserAction::Miss => Self::Miss,
                UserAction::Skip => Self::Skip,
                UserAction::Click => Self::Click,
            }
        }
    }

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
