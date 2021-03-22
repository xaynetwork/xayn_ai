use crate::{
    bert::Embedding,
    data::{document::DocumentId, CoiId},
};

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct DocumentIdComponent {
    pub id: DocumentId,
}

pub trait DocumentIdentifier {
    fn id(&self) -> &DocumentId;
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct DocumentContentComponent {
    pub snippet: String,
}

// TODO: the test-derived impls are temporarily available from rubert::utils::test_utils
#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct EmbeddingComponent {
    pub embedding: Embedding,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct LtrComponent {
    pub ltr_score: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct CoiComponent {
    /// The ID of the positive centre of interest
    pub id: CoiId,
    /// Distance from the positive centre of interest
    pub pos_distance: f32,
    /// Distance from the negative centre of interest
    pub neg_distance: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct ContextComponent {
    pub context_value: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct MabComponent {
    pub rank: usize,
}

// Document usage order:
// DocumentDataWithDocument -> DocumentDataWithEmbedding -> DocumentDataWithCoi ->
// DocumentDataWithLtr -> DocumentDataWithContext -> DocumentDataWithMab

pub struct DocumentDataWithDocument {
    pub document_id: DocumentIdComponent,
    pub document_content: DocumentContentComponent,
}

pub struct DocumentDataWithEmbedding {
    pub document_id: DocumentIdComponent,
    pub embedding: EmbeddingComponent,
}

impl DocumentIdentifier for DocumentDataWithEmbedding {
    fn id(&self) -> &DocumentId {
        &self.document_id.id
    }
}

impl DocumentDataWithEmbedding {
    pub fn from_document(
        document: DocumentDataWithDocument,
        embedding: EmbeddingComponent,
    ) -> Self {
        Self {
            document_id: document.document_id,
            embedding,
        }
    }
}

pub struct DocumentDataWithCoi {
    pub document_id: DocumentIdComponent,
    pub embedding: EmbeddingComponent,
    pub coi: CoiComponent,
}

impl DocumentDataWithCoi {
    pub fn from_document(document: DocumentDataWithEmbedding, coi: CoiComponent) -> Self {
        Self {
            document_id: document.document_id,
            embedding: document.embedding,
            coi,
        }
    }
}

pub struct DocumentDataWithLtr {
    pub document_id: DocumentIdComponent,
    pub embedding: EmbeddingComponent,
    pub coi: CoiComponent,
    pub ltr: LtrComponent,
}

impl DocumentDataWithLtr {
    pub fn from_document(document: DocumentDataWithCoi, ltr: LtrComponent) -> Self {
        Self {
            document_id: document.document_id,
            embedding: document.embedding,
            coi: document.coi,
            ltr,
        }
    }
}

#[cfg_attr(test, derive(Debug, Clone))]
pub struct DocumentDataWithContext {
    pub document_id: DocumentIdComponent,
    pub embedding: EmbeddingComponent,
    pub coi: CoiComponent,
    pub ltr: LtrComponent,
    pub context: ContextComponent,
}

impl DocumentDataWithContext {
    pub fn from_document(document: DocumentDataWithLtr, context: ContextComponent) -> Self {
        Self {
            document_id: document.document_id,
            embedding: document.embedding,
            coi: document.coi,
            ltr: document.ltr,
            context,
        }
    }
}

pub struct DocumentDataWithMab {
    pub document_id: DocumentIdComponent,
    pub embedding: EmbeddingComponent,
    pub coi: CoiComponent,
    pub ltr: LtrComponent,
    pub context: ContextComponent,
    pub mab: MabComponent,
}

impl DocumentIdentifier for DocumentDataWithMab {
    fn id(&self) -> &DocumentId {
        &self.document_id.id
    }
}

impl DocumentDataWithMab {
    pub fn from_document(document: DocumentDataWithContext, mab: MabComponent) -> Self {
        Self {
            document_id: document.document_id,
            embedding: document.embedding,
            coi: document.coi,
            ltr: document.ltr,
            context: document.context,
            mab,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::arr1;

    #[test]
    fn transition_and_get() {
        let document_id = DocumentIdComponent {
            id: DocumentId("id".to_string()),
        };
        let document_content = DocumentContentComponent {
            snippet: "snippet".to_string(),
        };
        let document_data = DocumentDataWithDocument {
            document_id: document_id.clone(),
            document_content: document_content.clone(),
        };
        assert_eq!(document_data.document_id, document_id);
        assert_eq!(document_data.document_content, document_content);

        let embedding = EmbeddingComponent {
            embedding: arr1(&[1., 2., 3., 4.]).into(),
        };
        let document_data =
            DocumentDataWithEmbedding::from_document(document_data, embedding.clone());
        assert_eq!(document_data.document_id, document_id);
        assert_eq!(document_data.embedding, embedding);

        let coi = CoiComponent {
            id: CoiId(9),
            pos_distance: 0.7,
            neg_distance: 0.2,
        };
        let document_data = DocumentDataWithCoi::from_document(document_data, coi.clone());
        assert_eq!(document_data.document_id, document_id);
        assert_eq!(document_data.embedding, embedding);
        assert_eq!(document_data.coi, coi);

        let ltr = LtrComponent { ltr_score: 0.3 };
        let document_data = DocumentDataWithLtr::from_document(document_data, ltr.clone());
        assert_eq!(document_data.document_id, document_id);
        assert_eq!(document_data.embedding, embedding);
        assert_eq!(document_data.coi, coi);
        assert_eq!(document_data.ltr, ltr);

        let context = ContextComponent {
            context_value: 1.23,
        };
        let document_data = DocumentDataWithContext::from_document(document_data, context.clone());
        assert_eq!(document_data.document_id, document_id);
        assert_eq!(document_data.embedding, embedding);
        assert_eq!(document_data.coi, coi);
        assert_eq!(document_data.ltr, ltr);
        assert_eq!(document_data.context, context);

        let mab = MabComponent { rank: 3 };
        let document_data = DocumentDataWithMab::from_document(document_data, mab.clone());
        assert_eq!(document_data.document_id, document_id);
        assert_eq!(document_data.embedding, embedding);
        assert_eq!(document_data.coi, coi);
        assert_eq!(document_data.ltr, ltr);
        assert_eq!(document_data.context, context);
        assert_eq!(document_data.mab, mab);
    }
}
