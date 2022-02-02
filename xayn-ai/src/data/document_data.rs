use serde::{Deserialize, Serialize};

use crate::{
    coi::CoiId,
    data::document::{Document, DocumentId, QueryId, SessionId},
    embedding::utils::Embedding,
    reranker::systems::CoiSystemData,
};

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct DocumentBaseComponent {
    pub(crate) id: DocumentId,
    pub(crate) initial_ranking: usize,
}

#[cfg_attr(test, derive(Debug, PartialEq, Default))]
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct DocumentContentComponent {
    pub(crate) title: String,
    pub(crate) snippet: String,
    pub(crate) session: SessionId,
    pub(crate) query_count: usize,
    pub(crate) query_id: QueryId,
    pub(crate) query_words: String,
    pub(crate) url: String,
    pub(crate) domain: String,
}

// TODO: the test-derived impls are temporarily available from rubert::utils::test_utils
#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct SMBertComponent {
    pub(crate) embedding: Embedding,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct QAMBertComponent {
    pub(crate) similarity: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
pub(crate) struct LtrComponent {
    pub(crate) ltr_score: f32,
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CoiComponent {
    /// The ID of the positive centre of interest
    pub(crate) id: CoiId,
    /// Similarity to the positive centre of interest
    pub(crate) pos_similarity: f32,
    /// Similarity to the negative centre of interest
    pub(crate) neg_similarity: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
pub(crate) struct ContextComponent {
    pub context_value: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
pub(crate) struct RankComponent {
    pub rank: usize,
}

/// Document usage order: [`DocumentDataWithDocument`]
/// -> [`DocumentDataWithSMBert`]
/// -> [`DocumentDataWithCoi`]
/// -> [`DocumentDataWithQAMBert`]
/// -> [`DocumentDataWithLtr`]
/// -> [`DocumentDataWithContext`]
/// -> [`DocumentDataWithRank`]
pub(crate) struct DocumentDataWithDocument {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) document_content: DocumentContentComponent,
}

impl From<&Document> for DocumentDataWithDocument {
    fn from(document: &Document) -> Self {
        Self {
            document_base: DocumentBaseComponent {
                id: document.id,
                initial_ranking: document.rank,
            },
            document_content: DocumentContentComponent {
                title: document.title.clone(),
                snippet: document.snippet.clone(),
                session: document.session,
                query_count: document.query_count,
                query_id: document.query_id,
                query_words: document.query_words.clone(),
                url: document.url.clone(),
                domain: document.domain.clone(),
            },
        }
    }
}

pub(crate) fn make_documents(documents: &[Document]) -> Vec<DocumentDataWithDocument> {
    documents.iter().map(Into::into).collect()
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct DocumentDataWithSMBert {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) document_content: DocumentContentComponent,
    pub(crate) smbert: SMBertComponent,
}

impl DocumentDataWithSMBert {
    pub(crate) fn from_document(
        document: &DocumentDataWithDocument,
        smbert: SMBertComponent,
    ) -> Self {
        Self {
            document_base: document.document_base.clone(),
            document_content: document.document_content.clone(),
            smbert,
        }
    }
}

impl CoiSystemData for DocumentDataWithSMBert {
    fn id(&self) -> DocumentId {
        self.document_base.id
    }

    fn smbert(&self) -> &SMBertComponent {
        &self.smbert
    }

    fn coi(&self) -> Option<&CoiComponent> {
        None
    }
}

#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct DocumentDataWithQAMBert {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) document_content: DocumentContentComponent,
    pub(crate) smbert: SMBertComponent,
    pub(crate) coi: CoiComponent,
    pub(crate) qambert: QAMBertComponent,
}

impl DocumentDataWithQAMBert {
    pub(crate) fn from_document(document: &DocumentDataWithCoi, qambert: QAMBertComponent) -> Self {
        Self {
            document_base: document.document_base.clone(),
            document_content: document.document_content.clone(),
            smbert: document.smbert.clone(),
            coi: document.coi.clone(),
            qambert,
        }
    }
}

pub(crate) struct DocumentDataWithCoi {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) document_content: DocumentContentComponent,
    pub(crate) smbert: SMBertComponent,
    pub(crate) coi: CoiComponent,
}

impl DocumentDataWithCoi {
    pub(crate) fn from_document(document: &DocumentDataWithSMBert, coi: CoiComponent) -> Self {
        Self {
            document_base: document.document_base.clone(),
            document_content: document.document_content.clone(),
            smbert: document.smbert.clone(),
            coi,
        }
    }
}

#[cfg_attr(test, derive(Debug))]
pub(crate) struct DocumentDataWithLtr {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) document_content: DocumentContentComponent,
    pub(crate) smbert: SMBertComponent,
    pub(crate) qambert: QAMBertComponent,
    pub(crate) coi: CoiComponent,
    pub(crate) ltr: LtrComponent,
}

impl DocumentDataWithLtr {
    pub(crate) fn from_document(document: &DocumentDataWithQAMBert, ltr: LtrComponent) -> Self {
        Self {
            document_base: document.document_base.clone(),
            document_content: document.document_content.clone(),
            smbert: document.smbert.clone(),
            qambert: document.qambert.clone(),
            coi: document.coi.clone(),
            ltr,
        }
    }
}

#[cfg_attr(test, derive(Debug, Clone))]
pub(crate) struct DocumentDataWithContext {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) document_content: DocumentContentComponent,
    pub(crate) smbert: SMBertComponent,
    pub(crate) qambert: QAMBertComponent,
    pub(crate) coi: CoiComponent,
    pub(crate) ltr: LtrComponent,
    pub(crate) context: ContextComponent,
}

impl DocumentDataWithContext {
    pub(crate) fn from_document(document: DocumentDataWithLtr, context: ContextComponent) -> Self {
        Self {
            document_base: document.document_base,
            document_content: document.document_content,
            smbert: document.smbert,
            qambert: document.qambert,
            coi: document.coi,
            ltr: document.ltr,
            context,
        }
    }
}

#[cfg_attr(test, derive(Clone, Debug, PartialEq))]
#[derive(Serialize, Deserialize)]
pub(crate) struct DocumentDataWithRank {
    pub(crate) document_base: DocumentBaseComponent,
    pub(crate) document_content: DocumentContentComponent,
    pub(crate) smbert: SMBertComponent,
    pub(crate) qambert: QAMBertComponent,
    pub(crate) coi: CoiComponent,
    pub(crate) ltr: LtrComponent,
    pub(crate) context: ContextComponent,
    pub(crate) rank: RankComponent,
}

impl DocumentDataWithRank {
    pub(crate) fn from_document(document: DocumentDataWithContext, rank: RankComponent) -> Self {
        Self {
            document_base: document.document_base,
            document_content: document.document_content,
            smbert: document.smbert,
            qambert: document.qambert,
            coi: document.coi,
            ltr: document.ltr,
            context: document.context,
            rank,
        }
    }
}

impl CoiSystemData for DocumentDataWithRank {
    fn id(&self) -> DocumentId {
        self.document_base.id
    }

    fn smbert(&self) -> &SMBertComponent {
        &self.smbert
    }

    fn coi(&self) -> Option<&CoiComponent> {
        Some(&self.coi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coi::CoiId;
    use ndarray::arr1;

    #[test]
    fn transition_and_get() {
        let document_id = DocumentBaseComponent {
            id: DocumentId::from_u128(0),
            initial_ranking: 23,
        };
        let document_content = DocumentContentComponent {
            title: "title".to_string(),
            snippet: "snippet".to_string(),
            query_words: "query".to_string(),
            ..Default::default()
        };
        let document_data = DocumentDataWithDocument {
            document_base: document_id.clone(),
            document_content: document_content.clone(),
        };
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.document_content, document_content);

        let embedding = SMBertComponent {
            embedding: arr1(&[1., 2., 3., 4.]).into(),
        };
        let document_data =
            DocumentDataWithSMBert::from_document(&document_data, embedding.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.smbert, embedding);

        let coi = CoiComponent {
            id: CoiId::mocked(9),
            pos_similarity: 0.7,
            neg_similarity: 0.2,
        };

        let document_data = DocumentDataWithCoi::from_document(&document_data, coi.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.smbert, embedding);
        assert_eq!(document_data.coi, coi);

        let qambert = QAMBertComponent { similarity: 0.5 };
        let document_data = DocumentDataWithQAMBert::from_document(&document_data, qambert.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.smbert, embedding);
        assert_eq!(document_data.qambert, qambert);

        let ltr = LtrComponent { ltr_score: 0.3 };
        let document_data = DocumentDataWithLtr::from_document(&document_data, ltr.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.smbert, embedding);
        assert_eq!(document_data.qambert, qambert);
        assert_eq!(document_data.coi, coi);
        assert_eq!(document_data.ltr, ltr);

        let context = ContextComponent {
            context_value: 1.23,
        };
        let document_data = DocumentDataWithContext::from_document(document_data, context.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.smbert, embedding);
        assert_eq!(document_data.qambert, qambert);
        assert_eq!(document_data.coi, coi);
        assert_eq!(document_data.ltr, ltr);
        assert_eq!(document_data.context, context);

        let rank = RankComponent { rank: 3 };
        let document_data = DocumentDataWithRank::from_document(document_data, rank.clone());
        assert_eq!(document_data.document_base, document_id);
        assert_eq!(document_data.smbert, embedding);
        assert_eq!(document_data.qambert, qambert);
        assert_eq!(document_data.coi, coi);
        assert_eq!(document_data.ltr, ltr);
        assert_eq!(document_data.context, context);
        assert_eq!(document_data.rank, rank);
    }
}
