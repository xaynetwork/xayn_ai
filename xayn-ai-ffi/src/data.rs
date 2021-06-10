use serde_repr::{Deserialize_repr, Serialize_repr};
use xayn_ai::RerankMode;

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
