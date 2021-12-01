use crate::{
    data::document_data::{ContextComponent, DocumentDataWithContext, DocumentDataWithLtr},
    reranker::systems::ContextSystem,
    Error,
};

/// Canonical `ContextSystem` instance.
pub(crate) struct Context;

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
                    doc.qambert.similarity,
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
    /// Average similarity distance.
    similarity_avg: f32,
}

impl ContextCalc {
    fn from_docs(docs: &[DocumentDataWithLtr]) -> Self {
        let docs_len = docs.len() as f32;
        let pos_avg = docs.iter().map(|doc| doc.coi.pos_distance).sum::<f32>() / docs_len;
        let neg_max = docs
            .iter()
            .map(|doc| doc.coi.neg_distance)
            .fold(f32::MIN, f32::max); // NOTE f32::max considers NaN as smallest value
        let similarity_avg = docs.iter().map(|doc| doc.qambert.similarity).sum::<f32>() / docs_len;

        Self {
            pos_avg,
            neg_max,
            similarity_avg,
        }
    }

    /// Calculates context value from given LTR score, positive distance, negative distance and similarity.
    fn calculate(&self, ltr_score: f32, pos: f32, neg: f32, similarity: f32) -> f32 {
        let frac_pos = (self.pos_avg > 0.)
            .then(|| (1. + pos / self.pos_avg).recip())
            .unwrap_or(1.);
        let frac_neg = (1. + (self.neg_max - neg)).recip();
        let frac_similarity = (self.similarity_avg > 0.)
            .then(|| (1. + similarity / self.similarity_avg).recip())
            .unwrap_or(0.5);

        // frac_similarity weights 70% all the others weight 10%
        (frac_similarity * 7. + frac_pos + frac_neg + ltr_score) / 10.
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;
    use crate::{
        coi::CoiId,
        data::{
            document::DocumentId,
            document_data::{
                CoiComponent,
                DocumentBaseComponent,
                DocumentContentComponent,
                LtrComponent,
                QAMBertComponent,
                SMBertComponent,
            },
        },
    };
    use test_utils::assert_approx_eq;

    struct LtrDocBuilder {
        docs: Vec<DocumentDataWithLtr>,
    }

    impl LtrDocBuilder {
        fn new() -> Self {
            Self { docs: vec![] }
        }

        fn add_doc(
            &mut self,
            ltr_score: f32,
            pos_distance: f32,
            neg_distance: f32,
            similarity: f32,
        ) {
            let id = DocumentId::from_u128(0);
            let embedding = arr1::<f32>(&[]).into();

            self.docs.push(DocumentDataWithLtr {
                document_base: DocumentBaseComponent {
                    id,
                    initial_ranking: 13,
                },
                document_content: DocumentContentComponent {
                    title: id.to_string(),
                    ..DocumentContentComponent::default()
                },
                smbert: SMBertComponent { embedding },
                qambert: QAMBertComponent { similarity },
                coi: CoiComponent {
                    id: CoiId::mocked(0),
                    pos_distance,
                    neg_distance,
                },
                ltr: LtrComponent { ltr_score },
            });
        }
    }

    #[test]
    fn test_calculate() {
        let calc = ContextCalc {
            pos_avg: 4.,
            neg_max: 8.,
            similarity_avg: 16.,
        };

        let cxt = calc.calculate(0., 0., calc.neg_max, 0.);
        assert_approx_eq!(f32, cxt, 9. / 10.); // 0 + 1/10 + 1/10 + 7/10

        let cxt = calc.calculate(1., 0., calc.neg_max, 0.);
        assert_approx_eq!(f32, cxt, 1.); // 1/10 + 1/10 + 1/10 + 7/10

        let cxt = calc.calculate(0., calc.pos_avg, calc.neg_max, calc.similarity_avg);
        assert_approx_eq!(f32, cxt, 50. / 100.); // 0 + 5/100 + 10/100 + 35/100

        let cxt = calc.calculate(0., 6., 7., 4.);
        assert_approx_eq!(f32, cxt, 65. / 100.); // 0 + 4/100 + 5/100 + 56/100
    }

    #[test]
    fn test_calculate_neg_max_f32_max() {
        // when calculating the negative distance in the `CoiSystem`,
        // we assign `f32::MAX` if we don't have negative cois
        let calc = ContextCalc {
            pos_avg: 4.,
            neg_max: f32::MAX,
            similarity_avg: 1.,
        };

        let ctx = calc.calculate(0., 0., calc.neg_max, 0.);
        assert_approx_eq!(f32, ctx, 9. / 10.); // 0 + 1/10 + 1/10 + 7/10
    }

    #[test]
    fn test_compute_from_docs() {
        let mut ltr_docs = LtrDocBuilder::new();
        ltr_docs.add_doc(0.9, 1., 10., 9.);
        ltr_docs.add_doc(0.5, 6., 4., 3.);
        ltr_docs.add_doc(0.3, 8., 2., 12.);

        let calc = ContextCalc::from_docs(&ltr_docs.docs);
        assert_approx_eq!(f32, calc.pos_avg, 5.);
        assert_approx_eq!(f32, calc.neg_max, 10.);
        assert_approx_eq!(f32, calc.similarity_avg, 8.);

        let cxt_docs = Context.compute_context(ltr_docs.docs);
        assert!(cxt_docs.is_ok());
        let cxt_docs = cxt_docs.unwrap();
        assert_eq!(cxt_docs.len(), 3);

        assert_approx_eq!(
            f32,
            cxt_docs[0].context.context_value,
            (5. / 6. + 1. + 0.9 + 8. / 17. * 7.) / 10.,
        );
        assert_approx_eq!(
            f32,
            cxt_docs[1].context.context_value,
            (5. / 11. + 1. / 7. + 0.5 + 8. / 11. * 7.) / 10.,
        );
        assert_approx_eq!(
            f32,
            cxt_docs[2].context.context_value,
            (5. / 13. + 1. / 9. + 0.3 + 8. / 20. * 7.) / 10.,
        );
    }
}
