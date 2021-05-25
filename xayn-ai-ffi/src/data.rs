use serde_repr::{Deserialize_repr, Serialize_repr};
use xayn_ai::{Relevance, UserFeedback};

/// A document relevance level.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Deserialize_repr, PartialEq, Serialize_repr)]
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

impl From<Relevance> for CRelevance {
    fn from(relevance: Relevance) -> Self {
        match relevance {
            Relevance::Low => Self::Low,
            Relevance::Medium => Self::Medium,
            Relevance::High => Self::High,
        }
    }
}

/// A user feedback level.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Deserialize_repr, PartialEq, Serialize_repr)]
pub enum CFeedback {
    Relevant = 0,
    Irrelevant = 1,
    // We can't use None nor Nil because they are reserved keywords in Dart or Objective-C.
    NotGiven = 2,
}

impl From<CFeedback> for UserFeedback {
    fn from(feedback: CFeedback) -> Self {
        match feedback {
            CFeedback::Relevant => Self::Relevant,
            CFeedback::Irrelevant => Self::Irrelevant,
            CFeedback::NotGiven => Self::None,
        }
    }
}

impl From<UserFeedback> for CFeedback {
    fn from(feedback: UserFeedback) -> Self {
        match feedback {
            UserFeedback::Relevant => Self::Relevant,
            UserFeedback::Irrelevant => Self::Irrelevant,
            UserFeedback::None => Self::NotGiven,
        }
    }
}
