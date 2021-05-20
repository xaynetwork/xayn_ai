use serde::{Deserialize, Serialize};

use crate::{
    data::{document::DocumentId, CoiId},
    reranker::systems::CoiSystemData,
    smbert::Embedding,
};

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
pub(crate) struct DocumentBaseComponent {
    pub(crate) id: DocumentId,
    pub(crate) initial_ranking: usize,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
pub(crate) struct DocumentContentComponent {
    pub(crate) snippet: String,
}

// TODO: the test-derived impls are temporarily available from rubert::utils::test_utils
#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct SMBertEmbeddingComponent {
    pub(crate) embedding: Embedding,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct QAMBertComponent {
    pub(crate) similarity: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
pub(crate) struct LtrComponent {
    pub(crate) ltr_score: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
pub(crate) struct CoiComponent {
    /// The ID of the positive centre of interest
    pub(crate) id: CoiId,
    /// Distance from the positive centre of interest
    pub(crate) pos_distance: f32,
    /// Distance from the negative centre of interest
    pub(crate) neg_distance: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
pub(crate) struct ContextComponent {
    pub context_value: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
pub(crate) struct MabComponent {
    pub rank: usize,
}

macro_rules! impl_coi_system_data_no_coi {
    ($type:ty) => {
        impl CoiSystemData for $type {
            fn id(&self) -> &DocumentId {
                &self.document_base.id
            }

            fn embedding(&self) -> &SMBertEmbeddingComponent {
                &self.embedding
            }

            fn coi(&self) -> Option<&CoiComponent> {
                None
            }
        }
    };
}

// Document usage order:
// DocumentDataWithDocument -> DocumentDataWithEmbedding -> DocumentDataWithCoi ->
// DocumentDataWithLtr -> DocumentDataWithContext -> DocumentDataWithMab

pub(crate) struct DocumentDataWithDocument {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) document_content: DocumentContentComponent,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct DocumentDataWithSMBert {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) document_content: DocumentContentComponent,
    pub(crate) embedding: SMBertEmbeddingComponent,
}

impl DocumentDataWithSMBert {
    pub(crate) fn from_document(
        document: DocumentDataWithDocument,
        embedding: SMBertEmbeddingComponent,
    ) -> Self {
        Self {
            document_base: document.document_base,
            document_content: document.document_content,
            embedding,
        }
    }
}

impl_coi_system_data_no_coi!(DocumentDataWithSMBert);

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct DocumentDataWithQAMBert {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) embedding: SMBertEmbeddingComponent,
    pub(crate) qambert: QAMBertComponent,
}

impl DocumentDataWithQAMBert {
    pub(crate) fn from_document(
        document: DocumentDataWithSMBert,
        qambert: QAMBertComponent,
    ) -> Self {
        Self {
            document_base: document.document_base,
            embedding: document.embedding,
            qambert,
        }
    }
}

impl_coi_system_data_no_coi!(DocumentDataWithQAMBert);

pub(crate) struct DocumentDataWithCoi {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) embedding: SMBertEmbeddingComponent,
    pub(crate) qambert: QAMBertComponent,
    pub(crate) coi: CoiComponent,
}

impl DocumentDataWithCoi {
    pub(crate) fn from_document(document: DocumentDataWithQAMBert, coi: CoiComponent) -> Self {
        Self {
            document_base: document.document_base,
            embedding: document.embedding,
            qambert: document.qambert,
            coi,
        }
    }
}

#[cfg_attr(test, derive(Debug))]
pub(crate) struct DocumentDataWithLtr {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) embedding: SMBertEmbeddingComponent,
    pub(crate) qambert: QAMBertComponent,
    pub(crate) coi: CoiComponent,
    pub(crate) ltr: LtrComponent,
}

impl DocumentDataWithLtr {
    pub(crate) fn from_document(document: DocumentDataWithCoi, ltr: LtrComponent) -> Self {
        Self {
            document_base: document.document_base,
            embedding: document.embedding,
            qambert: document.qambert,
            coi: document.coi,
            ltr,
        }
    }
}

#[cfg_attr(test, derive(Debug, Clone))]
pub(crate) struct DocumentDataWithContext {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) embedding: SMBertEmbeddingComponent,
    pub(crate) qambert: QAMBertComponent,
    pub(crate) coi: CoiComponent,
    pub(crate) ltr: LtrComponent,
    pub(crate) context: ContextComponent,
}

impl DocumentDataWithContext {
    pub(crate) fn from_document(document: DocumentDataWithLtr, context: ContextComponent) -> Self {
        Self {
            document_base: document.document_base,
            embedding: document.embedding,
            qambert: document.qambert,
            coi: document.coi,
            ltr: document.ltr,
            context,
        }
    }
}

#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
#[derive(Serialize, Deserialize)]
pub(crate) struct DocumentDataWithMab {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) embedding: SMBertEmbeddingComponent,
    pub(crate) qambert: QAMBertComponent,
    pub(crate) coi: CoiComponent,
    pub(crate) ltr: LtrComponent,
    pub(crate) context: ContextComponent,
    pub(crate) mab: MabComponent,
}

impl DocumentDataWithMab {
    pub(crate) fn from_document(document: DocumentDataWithContext, mab: MabComponent) -> Self {
        Self {
            document_base: document.document_base,
            embedding: document.embedding,
            qambert: document.qambert,
            coi: document.coi,
            ltr: document.ltr,
            context: document.context,
            mab,
        }
    }
}

impl CoiSystemData for DocumentDataWithMab {
    fn id(&self) -> &DocumentId {
        &self.document_base.id
    }

    fn embedding(&self) -> &SMBertEmbeddingComponent {
        &self.embedding
    }

    fn coi(&self) -> Option<&CoiComponent> {
        Some(&self.coi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn transition_and_get() {
        let document_id = DocumentBaseComponent {
            id: DocumentId::from_u128(0),
            initial_ranking: 23,
        };
        let document_content = DocumentContentComponent {
            snippet: "snippet".to_string(),
        };
        let document_data = DocumentDataWithDocument {
            document_base: document_id.clone(),
            document_content: document_content.clone(),
        };
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.document_content, document_content);

        let embedding = SMBertEmbeddingComponent {
            embedding: arr1(&[1., 2., 3., 4.]).into(),
        };
        let document_data = DocumentDataWithSMBert::from_document(document_data, embedding.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.embedding, embedding);

        let qambert = QAMBertComponent { similarity: 0.5 };
        let document_data = DocumentDataWithQAMBert::from_document(document_data, qambert.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.embedding, embedding);
        assert_eq!(document_data.qambert, qambert);

        let coi = CoiComponent {
            id: CoiId(9),
            pos_distance: 0.7,
            neg_distance: 0.2,
        };
        let document_data = DocumentDataWithCoi::from_document(document_data, coi.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.embedding, embedding);
        assert_eq!(document_data.qambert, qambert);
        assert_eq!(document_data.coi, coi);

        let ltr = LtrComponent { ltr_score: 0.3 };
        let document_data = DocumentDataWithLtr::from_document(document_data, ltr.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.embedding, embedding);
        assert_eq!(document_data.qambert, qambert);
        assert_eq!(document_data.coi, coi);
        assert_eq!(document_data.ltr, ltr);

        let context = ContextComponent {
            context_value: 1.23,
        };
        let document_data = DocumentDataWithContext::from_document(document_data, context.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.embedding, embedding);
        assert_eq!(document_data.qambert, qambert);
        assert_eq!(document_data.coi, coi);
        assert_eq!(document_data.ltr, ltr);
        assert_eq!(document_data.context, context);

        let mab = MabComponent { rank: 3 };
        let document_data = DocumentDataWithMab::from_document(document_data, mab.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.embedding, embedding);
        assert_eq!(document_data.qambert, qambert);
        assert_eq!(document_data.coi, coi);
        assert_eq!(document_data.ltr, ltr);
        assert_eq!(document_data.context, context);
        assert_eq!(document_data.mab, mab);
    }
}
