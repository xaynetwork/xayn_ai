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
    /// Relevance of the document
    pub relevance: Relevance,
    /// Swipe action of the user
    pub swipe_action: SwipeAction,
}

#[derive(Clone, Copy)]
pub enum Relevance {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy)]
pub enum SwipeAction {
    Relevant,
    Irrelevant,
}
