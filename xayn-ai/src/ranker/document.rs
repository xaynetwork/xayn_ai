// Copy of the document struct of the DE
use crate::{embedding::utils::Embedding, DocumentId};

pub trait Document {
    /// Gets the document id.
    fn id(&self) -> DocumentId;

    /// Gets the SMBert embedding of the document.
    fn smbert_embedding(&self) -> &Embedding;
}

#[cfg(test)]
pub(super) struct TestDocument {
    pub(super) id: DocumentId,
    pub(super) smbert_embedding: Embedding,
}

#[cfg(test)]
impl Document for TestDocument {
    fn id(&self) -> DocumentId {
        self.id
    }

    fn smbert_embedding(&self) -> &Embedding {
        &self.smbert_embedding
    }
}
