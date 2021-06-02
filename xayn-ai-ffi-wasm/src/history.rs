//! placeholder / later we can have a crate that contains common code for c-ffi and wasm

use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

use xayn_ai::{
    DayOfWeek,
    DocumentHistory,
    DocumentId,
    QueryId,
    Relevance,
    SessionId,
    UserAction,
    UserFeedback,
};

/// A document relevance level.
#[repr(u8)]
#[derive(Deserialize_repr, Serialize_repr)]
#[cfg_attr(test, derive(Clone))]
pub enum WRelevance {
    Low = 0,
    Medium = 1,
    High = 2,
}

impl From<WRelevance> for Relevance {
    fn from(relevance: WRelevance) -> Self {
        match relevance {
            WRelevance::Low => Self::Low,
            WRelevance::Medium => Self::Medium,
            WRelevance::High => Self::High,
        }
    }
}

/// A user feedback level.
#[repr(u8)]
#[derive(Deserialize_repr, Serialize_repr)]
#[cfg_attr(test, derive(Clone))]
pub enum WFeedback {
    Relevant = 0,
    Irrelevant = 1,
    // We cannot use None nor Nil because they are reserved
    // keyword in dart or objective-C
    NotGiven = 2,
}

impl From<WFeedback> for UserFeedback {
    fn from(feedback: WFeedback) -> Self {
        match feedback {
            WFeedback::Relevant => Self::Relevant,
            WFeedback::Irrelevant => Self::Irrelevant,
            WFeedback::NotGiven => Self::None,
        }
    }
}

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

#[derive(Deserialize, Serialize)]
pub struct WHistory {
    id: DocumentId,
    relevance: WRelevance,
    feedback: WFeedback,
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

    impl From<Relevance> for WRelevance {
        fn from(relevance: Relevance) -> Self {
            match relevance {
                Relevance::Low => Self::Low,
                Relevance::Medium => Self::Medium,
                Relevance::High => Self::High,
            }
        }
    }

    impl From<UserFeedback> for WFeedback {
        fn from(feedback: UserFeedback) -> Self {
            match feedback {
                UserFeedback::Relevant => Self::Relevant,
                UserFeedback::Irrelevant => Self::Irrelevant,
                UserFeedback::None => Self::NotGiven,
            }
        }
    }

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
