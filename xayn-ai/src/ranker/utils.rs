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

/// Represents a result from a query.
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique identifier of the document.
    pub id: Id,

    /// Embedding from smbert.
    pub smbert_embedding: Embedding,
}
