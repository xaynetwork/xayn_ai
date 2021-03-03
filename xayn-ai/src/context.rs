use crate::{
    data::document_data::{ContextComponent, DocumentDataWithContext, DocumentDataWithLtr},
    reranker_systems::ContextSystem,
    Error,
};

/// Canonical `ContextSystem` instance.
struct Context;

impl ContextSystem for Context {
    fn compute_context(
        &self,
        documents: Vec<DocumentDataWithLtr>,
    ) -> Result<Vec<DocumentDataWithContext>, Error> {
        let cxt_calc = ContextCalc::from_docs(&documents);
        Ok(documents
            .into_iter()
            .map(|doc| {
                let context_value = cxt_calc.calculate(
                    doc.ltr.ltr_score,
                    doc.coi.pos_distance,
                    doc.coi.neg_distance,
                );
                DocumentDataWithContext::from_document(doc, ContextComponent { context_value })
            })
            .collect())
    }
}

/// Calculator for context values.
struct ContextCalc {
    /// Average positive distance.
    pos_avg: f32,
    /// Maximum negative distance.
    neg_max: f32,
}

impl ContextCalc {
    fn from_docs(docs: &[DocumentDataWithLtr]) -> Self {
        let pos_avg = docs.iter().map(|doc| doc.coi.pos_distance).sum::<f32>() / docs.len() as f32;
        let neg_max = docs
            .iter()
            .map(|doc| doc.coi.neg_distance)
            .fold(f32::MIN, |curr_max, nd| curr_max.max(nd)); // NOTE ignores NaNs
        Self { pos_avg, neg_max }
    }

    /// Calculates context value from given LTR score, positive distance, negative distance.
    ///
    /// Age factor not currently implemented (assumed to be 1).
    fn calculate(&self, ltr_score: f32, pos: f32, neg: f32) -> f32 {
        let wt_div = 3f32;
        let frac_pos = (1f32 + pos / self.pos_avg).recip();
        let frac_neg = (1f32 + self.neg_max - neg).recip();

        (frac_pos + frac_neg + ltr_score) / wt_div
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        document::DocumentId,
        document_data::{CoiComponent, DocumentIdComponent, EmbeddingComponent, LtrComponent},
        CoiId,
    };

    struct LtrDocBuilder {
        docs: Vec<DocumentDataWithLtr>,
    }

    impl LtrDocBuilder {
        fn new() -> Self {
            Self { docs: vec![] }
        }

        fn add_doc(&mut self, ltr_score: f32, pos_distance: f32, neg_distance: f32) {
            let id = DocumentId("id".to_string());
            let embedding = vec![].into();

            self.docs.push(DocumentDataWithLtr {
                document_id: DocumentIdComponent { id },
                embedding: EmbeddingComponent { embedding },
                coi: CoiComponent {
                    id: CoiId(0),
                    pos_distance,
                    neg_distance,
                },
                ltr: LtrComponent { ltr_score },
            });
        }
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_calculate() {
        let calc = ContextCalc {
            pos_avg: 4.,
            neg_max: 8.,
        };
        let cxt = calc.calculate(0., 0., calc.neg_max);
        assert_eq!(cxt, 2. / 3.); // 1/3 + 1/3
        let cxt = calc.calculate(1., 0., calc.neg_max);
        assert_eq!(cxt, 1.); // 1/3 + 1/3 + 1/3
        let cxt = calc.calculate(0., calc.pos_avg, calc.neg_max);
        assert_eq!(cxt, 0.5); // 1/6 + 1/3
        let cxt = calc.calculate(0., 8., 7.);
        assert_eq!(cxt, 5. / 18.) // 1/9 + 1/6
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_compute_from_docs() {
        let mut ltr_docs = LtrDocBuilder::new();
        ltr_docs.add_doc(0.9, 1., 10.);
        ltr_docs.add_doc(0.5, 6., 4.);
        ltr_docs.add_doc(0.3, 8., 2.);
        // pos_avg = 5, neg_max = 10

        let cxt_docs = Context.compute_context(ltr_docs.docs);
        assert!(cxt_docs.is_ok());
        let cxt_docs = cxt_docs.unwrap();
        assert_eq!(cxt_docs.len(), 3);

        assert_eq!(cxt_docs[0].context.context_value, (5. / 6. + 1. + 0.9) / 3.);
        assert_eq!(
            cxt_docs[1].context.context_value,
            (5. / 11. + 1. / 7. + 0.5) / 3.
        );
        assert_eq!(
            cxt_docs[2].context.context_value,
            (5. / 13. + 1. / 9. + 0.3) / 3.
        );
    }
}
