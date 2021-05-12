use derive_more::Display;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use std::convert::TryFrom;

use crate::Error;

#[repr(transparent)]
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

/// This represents a result from the query.
#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    /// unique identifier of this document
    pub id: DocumentId,
    /// position of the document from the source
    pub rank: usize,
    pub snippet: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentHistory {
    /// unique identifier of this document
    pub id: DocumentId,
    /// Relevance level of the document
    pub relevance: Relevance,
    /// A flag that indicates whether the user liked the document
    pub user_feedback: UserFeedback,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum UserFeedback {
    Relevant,
    Irrelevant,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Relevance {
    Low,
    Medium,
    High,
}

/// The ranks are in the same logical order as the original documents.
pub type Ranks = Vec<usize>;
