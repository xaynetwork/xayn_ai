use crate::{
    data::{
        document::DocumentHistory,
        document_data::{DocumentDataWithCoi, DocumentDataWithLtr, LtrComponent},
    },
    error::Error,
    reranker::systems::LtrSystem,
};

/// LTR with constant value.
pub(crate) struct ConstLtr;

impl ConstLtr {
    const SCORE: f32 = 0.5;

    pub(crate) fn new() -> Self {
        // 0.5 is the only valid value.
        // It must be between 0 and 1. Since this is used to compute the context value
        // and context value is used to update `alpha` and `beta` of the cois.
        // Using a value different from 0.5 will change the parameters of a coi in an
        // umbalanced way.
        Self
    }
}

impl LtrSystem for ConstLtr {
    fn compute_ltr(
        &self,
        _history: &[DocumentHistory],
        documents: Vec<DocumentDataWithCoi>,
    ) -> Result<Vec<DocumentDataWithLtr>, Error> {
        let ltr_score = Self::SCORE;
        Ok(documents
            .into_iter()
            .map(|doc| DocumentDataWithLtr::from_document(doc, LtrComponent { ltr_score }))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use ndarray::arr1;

    use super::*;
    use crate::data::{
        document::DocumentId,
        document_data::{CoiComponent, DocumentBaseComponent, EmbeddingComponent},
        CoiId,
    };

    #[test]
    fn test_const_value() {
        let id = DocumentId("id1".to_string());
        let embedding = arr1(&[1., 2., 3., 4.]).into();
        let coi = CoiComponent {
            id: CoiId(9),
            pos_distance: 0.7,
            neg_distance: 0.2,
        };
        let doc1 = DocumentDataWithCoi {
            document_base: DocumentBaseComponent {
                id,
                initial_ranking: 24,
            },
            embedding: EmbeddingComponent { embedding },
            coi,
        };

        let id = DocumentId("id2".to_string());
        let embedding = arr1(&[5., 6., 7.]).into();
        let coi = CoiComponent {
            id: CoiId(5),
            pos_distance: 0.3,
            neg_distance: 0.9,
        };
        let doc2 = DocumentDataWithCoi {
            document_base: DocumentBaseComponent {
                id,
                initial_ranking: 42,
            },
            embedding: EmbeddingComponent { embedding },
            coi,
        };

        let res = ConstLtr::new().compute_ltr(&[], vec![doc1, doc2]);
        assert!(res.is_ok());
        let ltr_docs = res.unwrap();
        assert_eq!(ltr_docs.len(), 2);
        assert!(approx_eq!(f32, ltr_docs[0].ltr.ltr_score, 0.5));
        assert!(approx_eq!(f32, ltr_docs[1].ltr.ltr_score, 0.5));
    }
}
