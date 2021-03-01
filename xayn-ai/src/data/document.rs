#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct DocumentId(pub String);

// This represents a result from the query
pub struct Document {
    /// unique identifier of this document
    pub id: DocumentId,
    /// position of the document from the source
    pub rank: usize,
    pub snippet: String,
}

pub struct DocumentHistory {
    /// unique identifier of this document
    pub id: DocumentId,
    /// Relevance level of the document
    pub relevance: Relevance,
    /// A flag that indicates whether the user liked the document
    pub user_feedback: UserFeedback,
}

#[derive(Clone, Copy)]
pub enum UserFeedback {
    Relevant,
    Irrelevant,
    None,
}

#[derive(Clone, Copy)]
pub enum Relevance {
    Low,
    Medium,
    High,
}
