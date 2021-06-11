use serde_repr::{Deserialize_repr, Serialize_repr};
use xayn_ai::{DayOfWeek, Relevance, RerankMode, UserAction, UserFeedback};

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

/// Day of the week.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Deserialize_repr, PartialEq, Serialize_repr)]
pub enum CDayOfWeek {
    Mon = 0,
    Tue = 1,
    Wed = 2,
    Thu = 3,
    Fri = 4,
    Sat = 5,
    Sun = 6,
}

impl From<CDayOfWeek> for DayOfWeek {
    fn from(day: CDayOfWeek) -> Self {
        match day {
            CDayOfWeek::Mon => Self::Mon,
            CDayOfWeek::Tue => Self::Tue,
            CDayOfWeek::Wed => Self::Wed,
            CDayOfWeek::Thu => Self::Thu,
            CDayOfWeek::Fri => Self::Fri,
            CDayOfWeek::Sat => Self::Sat,
            CDayOfWeek::Sun => Self::Sun,
        }
    }
}

impl From<DayOfWeek> for CDayOfWeek {
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

/// Interaction from the user.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Deserialize_repr, PartialEq, Serialize_repr)]
pub enum CUserAction {
    Miss = 0,
    Skip = 1,
    Click = 2,
}

impl From<CUserAction> for UserAction {
    fn from(user_action: CUserAction) -> Self {
        match user_action {
            CUserAction::Miss => Self::Miss,
            CUserAction::Skip => Self::Skip,
            CUserAction::Click => Self::Click,
        }
    }
}

impl From<UserAction> for CUserAction {
    fn from(user_action: UserAction) -> Self {
        match user_action {
            UserAction::Miss => Self::Miss,
            UserAction::Skip => Self::Skip,
            UserAction::Click => Self::Click,
        }
    }
}

/// A document relevance level.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Deserialize_repr, PartialEq, Serialize_repr)]
pub enum CRerankMode {
    News = 0,
    Search = 1,
}

impl From<CRerankMode> for RerankMode {
    fn from(mode: CRerankMode) -> Self {
        match mode {
            CRerankMode::News => Self::News,
            CRerankMode::Search => Self::Search,
        }
    }
}

impl From<RerankMode> for CRerankMode {
    fn from(mode: RerankMode) -> Self {
        match mode {
            RerankMode::News => Self::News,
            RerankMode::Search => Self::Search,
        }
    }
}
