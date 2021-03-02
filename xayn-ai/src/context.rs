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
