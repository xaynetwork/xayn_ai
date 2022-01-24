// Copy of the document struct of the DE
use uuid::Uuid;

use crate::embedding::utils::Embedding;

/// Unique identifier of the [`Document`].
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Id(pub Uuid);

impl Id {
    /// Creates a [`Id`] from a 128bit value in big-endian order.
    pub fn from_u128(id: u128) -> Self {
        Id(Uuid::from_u128(id))
    }
}

pub trait Document {
    /// Gets the document id.
    fn id(&self) -> Id;

    /// Gets the SMBert embedding of the document.
    fn smbert_embedding(&self) -> &Embedding;
}

#[cfg(test)]
pub(super) struct TestDocument {
    pub(super) id: Id,
    pub(super) smbert_embedding: Embedding,
}

#[cfg(test)]
impl Document for TestDocument {
    fn id(&self) -> Id {
        self.id
    }

    fn smbert_embedding(&self) -> &Embedding {
        &self.smbert_embedding
    }
}
