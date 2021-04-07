#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct DocumentId(pub String);

impl From<&str> for DocumentId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

/// This represents a result from the query.
#[derive(Debug)]
pub struct Document {
    /// unique identifier of this document
    pub id: DocumentId,
    /// position of the document from the source
    pub rank: usize,
    pub snippet: String,
}

#[derive(Debug)]
pub struct DocumentHistory {
    /// unique identifier of this document
    pub id: DocumentId,
    /// Relevance level of the document
    pub relevance: Relevance,
    /// A flag that indicates whether the user liked the document
    pub user_feedback: UserFeedback,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UserFeedback {
    Relevant,
    Irrelevant,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Relevance {
    Low,
    Medium,
    High,
}
