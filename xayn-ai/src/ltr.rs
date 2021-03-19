use crate::{
    data::{
        document::DocumentHistory,
        document_data::{DocumentDataWithCoi, DocumentDataWithLtr, LtrComponent},
    },
    error::Error,
    reranker_systems::LtrSystem,
};

/// LTR with constant value.
struct ConstLtr(f32);

impl LtrSystem for ConstLtr {
    fn compute_ltr(
        &self,
        _history: &[DocumentHistory],
        documents: Vec<DocumentDataWithCoi>,
    ) -> Result<Vec<DocumentDataWithLtr>, Error> {
        let ltr_score = self.0;
        Ok(documents
            .into_iter()
            .map(|doc| DocumentDataWithLtr::from_document(doc, LtrComponent { ltr_score }))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;

    use super::*;
    use crate::{
        data::{
            document::DocumentId,
            document_data::{CoiComponent, DocumentIdComponent, EmbeddingComponent},
            CoiId,
        },
        ndarray::arr1,
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
            document_id: DocumentIdComponent { id },
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
            document_id: DocumentIdComponent { id },
            embedding: EmbeddingComponent { embedding },
            coi,
        };

        let half = 0.5;
        let res = ConstLtr(half).compute_ltr(&[], vec![doc1, doc2]);
        assert!(res.is_ok());
        let ltr_docs = res.unwrap();
        assert_eq!(ltr_docs.len(), 2);
        assert!(approx_eq!(f32, ltr_docs[0].ltr.ltr_score, half));
        assert!(approx_eq!(f32, ltr_docs[1].ltr.ltr_score, half));
    }
}
