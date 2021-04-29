#![allow(dead_code)] // TEMP

use crate::ltr::features::dataiku::{
    click_entropy,
    mean_recip_rank,
    ClickSat,
    Query,
    SearchResult,
};
use itertools::Itertools;
use std::collections::HashSet;

struct QueryFeatures {
    /// Entropy over ranks of clicked results.
    click_entropy: f32,
    /// Number of terms.
    num_terms: usize,
    /// Average `n` where query is the `n`th of a session.
    rank_per_session: f32,
    /// Average number of occurrences per session.
    occurs_per_session: f32,
    /// Total number of occurrences.
    num_occurs: usize,
    /// Mean reciprocal rank of clicked results.
    click_mrr: f32,
    /// Average number of clicks.
    avg_clicks: f32,
    /// Average number of skips.
    avg_skips: f32,
}

/// Calculate query features for the given query and historical search results of a user.
fn query_features(history: &[SearchResult], query: Query) -> QueryFeatures {
    // history filtered by query
    let history_q = history
        .iter()
        .filter(|r| r.query_id == query.id)
        .collect_vec();

    let click_entropy = click_entropy(&history_q);

    let (rank_per_session, num_occurs) = avg_query_count(history_q.iter());

    let num_sessions = history_q.iter().unique_by(|r| r.session_id).count() as f32;
    let occurs_per_session = num_occurs as f32 / num_sessions;

    let clicked = history_q
        .iter()
        .filter(|r| r.relevance > ClickSat::Low)
        .collect_vec();
    let click_mrr = mean_recip_rank(&clicked, None, None);

    let avg_clicks = clicked.len() as f32 / num_occurs as f32;

    let avg_skips = history_q
        .into_iter()
        .filter(|r| r.relevance == ClickSat::Skip)
        .count() as f32
        / num_occurs as f32;

    QueryFeatures {
        click_entropy,
        num_terms: query.words.len(),
        rank_per_session,
        occurs_per_session,
        num_occurs,
        click_mrr,
        avg_clicks,
        avg_skips,
    }
}

/// Calculate average `n` where query is the `n`th of a session.
/// Also returns the total number of searches the average is taken over.
fn avg_query_count<'a>(history_q: impl Iterator<Item = &'a &'a SearchResult>) -> (f32, usize) {
    let occurs = history_q
        .map(|r| (r.session_id, r.query_counter))
        .collect::<HashSet<_>>();

    let num_occurs = occurs.len();

    let rank_sum = occurs
        .into_iter()
        .map(|(_, query_count)| query_count as f32)
        .sum::<f32>();
    let rank_per_session = rank_sum / num_occurs as f32;

    (rank_per_session, num_occurs)
}
