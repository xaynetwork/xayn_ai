//! placeholder / later we can have a crate that contains common code for c-ffi and wasm

use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

use xayn_ai::{DocumentHistory, DocumentId, Relevance, UserFeedback};

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

#[derive(Deserialize, Serialize)]
pub struct WHistory {
    id: DocumentId,
    relevance: WRelevance,
    feedback: WFeedback,
}

impl From<WHistory> for DocumentHistory {
    fn from(history: WHistory) -> Self {
        Self {
            id: history.id,
            relevance: history.relevance.into(),
            user_feedback: history.feedback.into(),
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

    impl From<DocumentHistory> for WHistory {
        fn from(history: DocumentHistory) -> Self {
            Self {
                id: history.id,
                relevance: history.relevance.into(),
                feedback: history.user_feedback.into(),
            }
        }
    }
}
