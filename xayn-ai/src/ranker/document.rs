use chrono::NaiveDateTime;

use crate::{embedding::utils::Embedding, DocumentId};

pub trait Document {
    /// Gets the document id.
    fn id(&self) -> DocumentId;

    /// Gets the SMBert embedding of the document.
    fn smbert_embedding(&self) -> &Embedding;

    /// Gets the publishing date.
    fn date_published(&self) -> NaiveDateTime;
}

#[cfg(test)]
pub(super) struct TestDocument {
    pub(super) id: DocumentId,
    pub(super) smbert_embedding: Embedding,
    pub(super) date_published: NaiveDateTime,
}

#[cfg(test)]
impl TestDocument {
    pub(super) fn new(id: u128, embedding: impl Into<Embedding>, date_published: &str) -> Self {
        Self {
            id: DocumentId::from_u128(id),
            smbert_embedding: embedding.into(),
            date_published: NaiveDateTime::parse_from_str(date_published, "%Y-%m-%d %H:%M:%S")
                .unwrap(),
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

    fn date_published(&self) -> NaiveDateTime {
        self.date_published
    }
}
