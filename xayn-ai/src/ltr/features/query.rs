use super::{click_entropy, mean_recip_rank, DocSearchResult, HistSearchResult};
use itertools::Itertools;
use std::collections::HashSet;

/// Features specific to a given query.
pub(super) struct QueryFeatures {
    /// Entropy over ranks of clicked results.
    pub(super) click_entropy: f32,
    /// Number of terms.
    pub(super) num_terms: usize,
    /// Average `n` where query is the `n`th of a session.
    pub(super) mean_query_count: f32,
    /// Average number of occurrences per session.
    pub(super) occurs_per_session: f32,
    /// Total number of occurrences.
    pub(super) num_occurs: usize,
    /// Mean reciprocal rank of clicked results.
    pub(super) click_mrr: f32,
    /// Average number of clicks.
    pub(super) mean_clicks: f32,
    /// Average number of skips.
    pub(super) mean_skips: f32,
}

impl QueryFeatures {
    /// Build query features for the given search result and history of a user.
    pub(super) fn build(hists: &[HistSearchResult], doc: &DocSearchResult) -> Self {
        // history filtered by query
        let hists_q = hists
            .iter()
            .filter(|hist| hist.query.query_id == doc.query.query_id)
            .collect_vec();

        let num_terms = doc.query.query_words.len();

        if hists_q.is_empty() {
            return Self {
                click_entropy: 0.,
                num_terms,
                mean_query_count: 0.,
                occurs_per_session: 0.,
                num_occurs: 0,
                click_mrr: 0.283,
                mean_clicks: 0.,
                mean_skips: 0.,
            };
        }

        let click_entropy = click_entropy(&hists_q);

        let (mean_query_count, num_occurs) = mean_query_count(hists_q.iter());

        let num_sessions = hists_q
            .iter()
            .unique_by(|hist| hist.query.session_id)
            .count() as f32;
        let occurs_per_session = num_occurs as f32 / num_sessions;

        let clicked = hists_q
            .iter()
            .filter(|hist| hist.is_clicked())
            .collect_vec();
        let click_mrr = mean_recip_rank(&clicked, None, None);

        let mean_clicks = clicked.len() as f32 / num_occurs as f32;

        let mean_skips =
            hists_q.into_iter().filter(|hist| hist.is_skipped()).count() as f32 / num_occurs as f32;

        Self {
            click_entropy,
            num_terms,
            mean_query_count,
            occurs_per_session,
            num_occurs,
            click_mrr,
            mean_clicks,
            mean_skips,
        }
    }
}

/// Calculate average `n` where query is the `n`th of a session.
/// Also returns the total number of searches the average is taken over.
fn mean_query_count<'a>(history_q: impl Iterator<Item = &'a &'a HistSearchResult>) -> (f32, usize) {
    let occurs = history_q
        .map(|hist| (hist.query.session_id, hist.query.query_count))
        .collect::<HashSet<_>>();

    let num_occurs = occurs.len();

    let sum_query_count = occurs
        .into_iter()
        .map(|(_, query_count)| query_count as f32)
        .sum::<f32>();
    let mean_query_count = sum_query_count / num_occurs as f32;

    (mean_query_count, num_occurs)
}
