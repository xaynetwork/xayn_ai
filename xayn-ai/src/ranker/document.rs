// Copy of the document struct of the DE
use crate::{embedding::utils::Embedding, DocumentId};

pub trait Document {
    /// Gets the document id.
    fn id(&self) -> DocumentId;

    /// Gets the SMBert embedding of the document.
    fn smbert_embedding(&self) -> &Embedding;

    /// Get the API score.
    fn score(&self) -> Option<f32>;

    /// Get the API rank.
    fn rank(&self) -> usize;
}

#[cfg(test)]
pub(super) struct TestDocument {
    pub(super) id: DocumentId,
    pub(super) smbert_embedding: Embedding,
    pub(super) score: Option<f32>,
    pub(super) rank: usize,
}

#[cfg(test)]
impl TestDocument {
    pub(super) fn new(
        id: u128,
        embedding: impl Into<Embedding>,
        score: Option<f32>,
        rank: usize,
    ) -> Self {
        Self {
            id: DocumentId::from_u128(id),
            smbert_embedding: embedding.into(),
            score,
            rank,
        }
    }
}

#[cfg(test)]
impl Document for TestDocument {
    fn id(&self) -> DocumentId {
        self.id
    }

    fn smbert_embedding(&self) -> &Embedding {
        &self.smbert_embedding
    }

    /// Get the API score.
    fn score(&self) -> Option<f32> {
        self.score
    }

    /// Get the API rank.
    fn rank(&self) -> usize {
        self.rank
    }
}
