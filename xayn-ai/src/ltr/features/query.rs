#![allow(dead_code)] // TEMP

use super::{click_entropy, mean_recip_rank, ClickSat, SearchResult};
use itertools::Itertools;
use std::collections::HashSet;

/// Features specific to a given query.
pub(crate) struct QueryFeatures {
    /// Entropy over ranks of clicked results.
    click_entropy: f32,
    /// Number of terms.
    num_terms: usize,
    /// Average `n` where query is the `n`th of a session.
    mean_query_count: f32,
    /// Average number of occurrences per session.
    occurs_per_session: f32,
    /// Total number of occurrences.
    num_occurs: usize,
    /// Mean reciprocal rank of clicked results.
    click_mrr: f32,
    /// Average number of clicks.
    mean_clicks: f32,
    /// Average number of skips.
    mean_skips: f32,
}

impl QueryFeatures {
    /// Build query features for the given search result and history of a user.
    pub(crate) fn build(history: &[SearchResult], res: &SearchResult) -> Self {
        // history filtered by query
        let history_q = history
            .iter()
            .filter(|r| r.query_id == res.query_id)
            .collect_vec();

        let num_terms = res.query_words.len();

        if history_q.is_empty() {
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

        let click_entropy = click_entropy(&history_q);

        let (mean_query_count, num_occurs) = mean_query_count(history_q.iter());

        let num_sessions = history_q.iter().unique_by(|r| r.session_id).count() as f32;
        let occurs_per_session = num_occurs as f32 / num_sessions;

        let clicked = history_q
            .iter()
            .filter(|r| r.relevance > ClickSat::Low)
            .collect_vec();
        let click_mrr = mean_recip_rank(&clicked, None, None);

        let mean_clicks = clicked.len() as f32 / num_occurs as f32;

        let mean_skips = history_q
            .into_iter()
            .filter(|r| r.relevance == ClickSat::Skip)
            .count() as f32
            / num_occurs as f32;

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
fn mean_query_count<'a>(history_q: impl Iterator<Item = &'a &'a SearchResult>) -> (f32, usize) {
    let occurs = history_q
        .map(|r| (r.session_id, r.query_counter))
        .collect::<HashSet<_>>();

    let num_occurs = occurs.len();

    let sum_query_count = occurs
        .into_iter()
        .map(|(_, query_count)| query_count as f32)
        .sum::<f32>();
    let mean_query_count = sum_query_count / num_occurs as f32;

    (mean_query_count, num_occurs)
}
