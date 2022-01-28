use std::collections::HashMap;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};
use std::convert::TryFrom;
use uuid::Uuid;

use crate::{
    reranker::{systems::CoiSystemData, RerankMode},
    Error,
};

use super::document_data::DocumentDataWithRank;

#[repr(transparent)]
#[cfg_attr(test, derive(Default))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, Display)]
#[serde(transparent)]
pub struct DocumentId(pub Uuid);

impl DocumentId {
    //// Creates a DocumentId from a 128bit value in big-endian order.
    pub fn from_u128(id: u128) -> Self {
        DocumentId(Uuid::from_u128(id))
    }
}

impl TryFrom<&str> for DocumentId {
    type Error = Error;

    fn try_from(id: &str) -> Result<Self, Self::Error> {
        Ok(DocumentId(Uuid::parse_str(id)?))
    }
}

#[repr(transparent)]
#[cfg_attr(test, derive(Default))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, Display)]
#[serde(transparent)]
pub struct SessionId(pub Uuid);

impl SessionId {
    /// New identifier from a 128bit value in big-endian order.
    pub fn from_u128(id: u128) -> Self {
        Self(Uuid::from_u128(id))
    }
}

impl TryFrom<&str> for SessionId {
    type Error = Error;

    fn try_from(id: &str) -> Result<Self, Self::Error> {
        Ok(Self(Uuid::parse_str(id)?))
    }
}

#[repr(transparent)]
#[cfg_attr(test, derive(Default))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, Display)]
#[serde(transparent)]
pub struct QueryId(pub Uuid);

impl QueryId {
    /// New identifier from a 128bit value in big-endian order.
    pub fn from_u128(id: u128) -> Self {
        Self(Uuid::from_u128(id))
    }
}

impl TryFrom<&str> for QueryId {
    type Error = Error;

    fn try_from(id: &str) -> Result<Self, Self::Error> {
        Ok(Self(Uuid::parse_str(id)?))
    }
}

/// Represents a result from a query.
#[cfg_attr(test, derive(Default))]
#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier of the document
    pub id: DocumentId,
    /// Position of the document from the source
    pub rank: usize,
    /// Text title of the document
    pub title: String,
    /// Text snippet of the document
    pub snippet: String,
    /// Session of the document
    pub session: SessionId,
    /// Query count within session
    pub query_count: usize,
    /// Query identifier of the document
    pub query_id: QueryId,
    /// Query of the document
    pub query_words: String,
    /// URL of the document
    pub url: String,
    /// Domain of the document
    pub domain: String,
}

/// Represents a historical result from a query.
#[cfg_attr(test, derive(Default))]
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DocumentHistory {
    /// Unique identifier of the document
    pub id: DocumentId,
    /// Relevance level of the document
    pub relevance: Relevance,
    /// A flag that indicates whether the user liked the document
    pub user_feedback: UserFeedback,
    /// Session of the document
    pub session: SessionId,
    /// Query count within session
    pub query_count: usize,
    /// Query identifier of the document
    pub query_id: QueryId,
    /// Query of the document
    pub query_words: String,
    /// Day of week query was performed
    pub day: DayOfWeek,
    /// URL of the document
    pub url: String,
    /// Domain of the document
    pub domain: String,
    /// Reranked position of the document
    pub rank: usize,
    /// User interaction for the document
    pub user_action: UserAction,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum UserFeedback {
    Relevant = 0,
    Irrelevant = 1,
    NotGiven = 2,
}

impl Default for UserFeedback {
    fn default() -> Self {
        Self::NotGiven
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum Relevance {
    Low = 0,
    Medium = 1,
    High = 2,
}

/// The outcome of running the reranker.
///
/// This is named outcome instead of result as rust uses the
/// word result in a very specific way which doesn't apply here.
///
/// Besides the `final_ranking` all other fields are optional
/// as depending on configurations (don't run QA-mBERT in certain
/// context) and errors they might not be available.
///
/// Like `final_ranking` already did in the past they match this
/// information to the input documents by their index.
#[derive(Serialize, Deserialize)]
pub struct RerankingOutcomes {
    /// The final ranking computed by the AI.
    /// If the AI cannot run till having a final rank for each document the
    /// initial rank will be used here.
    pub final_ranking: Vec<u16>,

    /// The QA-mBERT outcomes (similarities)
    pub qambert_similarities: Option<Vec<f32>>,

    /// The context score of each document.
    ///
    /// It can be `None` if it was impossible to compute them.
    pub context_scores: Option<Vec<f32>>,
}

impl RerankingOutcomes {
    /// Creates a `RerankingOutcome` which contains all information.
    pub(crate) fn from_rank(
        mode: RerankMode,
        docs: &[Document],
        docs_with_rank: &[DocumentDataWithRank],
    ) -> Self {
        let docs_with_rank = docs_with_rank
            .iter()
            .map(|doc| (doc.id(), doc))
            .collect::<HashMap<_, _>>();

        let final_ranking = docs
            .iter()
            .map(|doc| docs_with_rank[&doc.id].rank.rank as u16)
            .collect();
        let qambert_similarities = mode.is_search().then(|| {
            docs.iter()
                .map(|doc| docs_with_rank[&doc.id].qambert.similarity)
                .collect()
        });
        let context_scores = mode.is_personalized().then(|| {
            docs.iter()
                .map(|doc| docs_with_rank[&doc.id].context.context_value)
                .collect()
        });

        Self {
            final_ranking,
            qambert_similarities,
            context_scores,
        }
    }

    /// Creates a `RerankingOutcome` from the initial ranking.
    ///
    /// This is used if reranking failed.
    pub(crate) fn from_initial_ranking(docs: &[Document]) -> Self {
        Self {
            final_ranking: docs.iter().map(|doc| doc.rank as u16).collect(),
            qambert_similarities: None,
            context_scores: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum UserAction {
    Miss = 0,
    Skip = 1,
    Click = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum DayOfWeek {
    Mon = 0,
    Tue = 1,
    Wed = 2,
    Thu = 3,
    Fri = 4,
    Sat = 5,
    Sun = 6,
}

impl DayOfWeek {
    /// Crates a `DayOfWeek` based on a wrap-around offset from `Mon`.
    pub fn from_day_offset(day_offset: usize) -> DayOfWeek {
        use DayOfWeek::*;
        static DAYS: &[DayOfWeek] = &[Mon, Tue, Wed, Thu, Fri, Sat, Sun];
        DAYS[day_offset % 7]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Default for Relevance {
        fn default() -> Self {
            Self::Low
        }
    }

    impl Default for UserAction {
        fn default() -> Self {
            Self::Miss
        }
    }

    impl Default for DayOfWeek {
        fn default() -> Self {
            DayOfWeek::Mon
        }
    }
}
