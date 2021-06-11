use derive_more::Display;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use uuid::Uuid;

use crate::{
    reranker::{systems::CoiSystemData, RerankMode},
    Error,
};

use super::document_data::DocumentDataWithMab;

#[repr(transparent)]
#[cfg_attr(test, derive(Default))]
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize, Display)]
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
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize, Display)]
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
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize, Display)]
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
#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum UserFeedback {
    Relevant,
    Irrelevant,
    None,
}

impl Default for UserFeedback {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Relevance {
    Low,
    Medium,
    High,
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
    /// The final ranking.
    ///
    /// If everything succeeds this is based on the result of the
    /// `MAB` step (which is based on the merged ranking scores
    /// of the other parts in the pipeline).
    ///
    /// But if various steps fail this might be based on something
    /// else, in the extreme case this is just the initial ranking.
    ///
    /// Make sure to check for the errors from the `Reranker`.
    pub final_ranking: Vec<u16>,

    /// The QA-mBERT outcomes (similarities)
    pub qa_mbert_similarities: Option<Vec<f32>>,

    /// The context score(s) which where feet into `MAB`.
    ///
    /// If due to errors no context scores are produced this
    /// is `None`.
    pub context_scores: Option<Vec<f32>>,
}

impl RerankingOutcomes {
    /// Creates a `RerankingOutcome` which contains all information.
    pub(crate) fn from_mab(
        mode: RerankMode,
        docs: &[Document],
        docs_with_mab: &[DocumentDataWithMab],
    ) -> Self {
        let docs_with_mab = docs_with_mab
            .iter()
            .map(|doc| (doc.id(), doc))
            .collect::<HashMap<_, _>>();

        let docs_len = docs.len();
        let mut final_ranking = Vec::with_capacity(docs_len);
        let mut context_scores = Vec::with_capacity(docs_len);
        let mut qa_mbert_similarities =
            matches!(mode, RerankMode::Search).then(|| Vec::with_capacity(docs_len));

        for doc in docs {
            let data = docs_with_mab[&doc.id];
            final_ranking.push(data.mab.rank as u16);
            context_scores.push(data.context.context_value);
            if let Some(vs) = qa_mbert_similarities.as_mut() {
                vs.push(data.qambert.similarity)
            }
        }

        Self {
            final_ranking,
            context_scores: Some(context_scores),
            qa_mbert_similarities,
        }
    }

    /// Creates a `RerankingOutcome` from the initial ranking.
    ///
    /// This is used if reranking failed.
    pub(crate) fn from_initial_ranking(docs: &[Document]) -> Self {
        Self {
            final_ranking: docs.iter().map(|doc| doc.rank as u16).collect(),
            qa_mbert_similarities: None,
            context_scores: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum UserAction {
    Click,
    Skip,
    Miss,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DayOfWeek {
    Mon,
    Tue,
    Wed,
    Thu,
    Fri,
    Sat,
    Sun,
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
