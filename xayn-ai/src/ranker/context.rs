use crate::{data::document_data::CoiComponent, ranker::util::Id};

pub(crate) struct Context {
    /// Average positive distance.
    pos_avg: f32,
    /// Maximum negative distance.
    neg_max: f32,
}

impl Context {
    pub(crate) fn from_cois(cois: &[(Id, CoiComponent)]) -> Self {
        let cois_len = cois.len() as f32;
        let pos_avg = cois.iter().map(|(_, coi)| coi.pos_distance).sum::<f32>() / cois_len;
        let neg_max = cois
            .iter()
            .map(|(_, coi)| coi.neg_distance)
            .fold(f32::MIN, f32::max); // NOTE f32::max considers NaN as smallest value

        Self { pos_avg, neg_max }
    }

    /// Calculates context value from given LTR score, positive distance, negative distance and similarity.
    pub(crate) fn calculate(&self, pos: f32, neg: f32) -> f32 {
        let frac_pos = (self.pos_avg > 0.)
            .then(|| (5. + pos / self.pos_avg).recip())
            .unwrap_or(5.);
        let frac_neg = (5. + (self.neg_max - neg)).recip();

        (frac_pos + frac_neg) / 10.
    }
}
