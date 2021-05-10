//! placeholder / later we can have a crate that contains common code for c-ffi and wasm

use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

use xayn_ai::{DocumentHistory, DocumentId, Relevance, UserFeedback};

/// A document relevance level.
#[repr(u8)]
#[derive(Deserialize_repr, Serialize_repr)]
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
