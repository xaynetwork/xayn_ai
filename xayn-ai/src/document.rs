#[repr(transparent)]
#[derive(Debug, PartialEq, Clone)]
pub struct DocumentId(pub String);

// This represent a result from the query
pub struct Document {
    /// unique identifier of this document
    pub id: DocumentId,
    /// position of the document from the source
    pub rank: usize,
    pub snippet: String,
}
