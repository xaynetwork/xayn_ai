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

pub(super) struct QueryFeatures {
    /// Entropy over ranks of clicked results.
    pub(super) click_entropy: f32,
    /// Number of terms.
    pub(super) num_terms: usize,
    /// Average `n` where query is the `n`th of a session.
    pub(super) mean_query_counter: f32,
    /// Average number of occurrences per session.
    pub(super) mean_occurs_per_session: f32,
    /// Total number of occurrences.
    pub(super) num_occurs: usize,
    /// Mean reciprocal rank of clicked results.
    pub(super) click_mrr: f32,
    /// Average number of clicks.
    pub(super) mean_clicks: f32,
    /// Average number of skips.
    pub(super) mean_non_click: f32,
}

impl QueryFeatures {
    pub(super) fn exact(history: &[SearchResult], query: &Query) -> QueryFeatures {
        //FIXME temp. to make reviews easier by not showing the whole `query_features` function as changed
        query_features(history, query)
    }
}

/// Calculate query features for the given query and historical search results of a user.
pub(super) fn query_features(history: &[SearchResult], query: &Query) -> QueryFeatures {
    let num_terms = query.words.len();

    // history filtered by query
    let history_q = history
        .iter()
        .filter(|r| r.query_id == query.id)
        .collect_vec();

    if history_q.is_empty() {
        return QueryFeatures {
            click_entropy: 0.0,
            num_terms,
            mean_query_counter: 0.0,
            mean_occurs_per_session: 0.0,
            num_occurs: 0,
            click_mrr: 0.283,
            mean_clicks: 0.0,
            mean_non_click: 0.0,
        };
    }

    let click_entropy = click_entropy(&history_q);

    // Soundgarden calls this `query_avg_position` but calculates the
    // mean/average of the `query_counter`! values in `history_q`.
    let (mean_query_counter, num_occurs) = mean_query_count(history_q.iter());

    let num_sessions = history_q.iter().unique_by(|r| r.session_id).count() as f32;
    let mean_occurs_per_session = num_occurs as f32 / num_sessions;

    let clicked = history_q
        .iter()
        .filter(|r| r.relevance > ClickSat::Low)
        .collect_vec();
    let click_mrr = mean_recip_rank(&clicked, None, None);

    let mean_clicks = clicked.len() as f32 / num_occurs as f32;

    let mean_non_click = history_q
        .into_iter()
        .filter(|r| r.relevance < ClickSat::Medium)
        .count() as f32
        / num_occurs as f32;

    QueryFeatures {
        click_entropy,
        num_terms,
        mean_query_counter,
        mean_occurs_per_session,
        num_occurs,
        click_mrr,
        mean_clicks,
        mean_non_click,
    }
}

/// Calculate average `n` where query is the `n`th of a session.
/// Also returns the total number of searches the average is taken over.
fn mean_query_count<'a>(history_q: impl Iterator<Item = &'a &'a SearchResult>) -> (f32, usize) {
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

#[cfg(test)]
mod tests {
    use crate::ltr::features::dataiku::{DayOfWeek, Rank};

    use super::*;

    #[test]
    fn test_query_features_no_matching_history() {
        let QueryFeatures {
            click_entropy,
            num_terms,
            mean_query_counter,
            mean_occurs_per_session,
            num_occurs,
            click_mrr,
            mean_clicks,
            mean_non_click,
        } = QueryFeatures::exact(
            &[],
            &Query {
                id: 233,
                words: vec![2, 100, 4],
            },
        );

        assert_approx_eq!(f32, click_entropy, 0.0);
        assert_eq!(num_terms, 3);

        assert_approx_eq!(f32, mean_query_counter, 0.0);
        assert_approx_eq!(f32, mean_occurs_per_session, 0.0);
        assert_eq!(num_occurs, 0);

        assert_approx_eq!(f32, click_mrr, 0.283);
        assert_approx_eq!(f32, mean_clicks, 0.0);
        assert_approx_eq!(f32, mean_non_click, 0.0);
    }

    fn history<'a>(
        query_id: i32,
        iter: impl IntoIterator<Item = &'a (i32, ClickSat)>,
    ) -> Vec<SearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (domain, relevance))| {
                let id = id as i32;
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                SearchResult {
                    session_id: 1,
                    user_id: 1,
                    query_id,
                    day: DayOfWeek::Tue,
                    query_words: vec![1, 2, id],
                    url: in_query_id,
                    domain: *domain,
                    relevance: *relevance,
                    position: Rank::from_usize(1 + in_query_id as usize),
                    query_counter: per_query_id as u8,
                }
            })
            .collect()
    }

    #[test]
    fn test_query_features() {
        let history = history(
            423,
            &[
                /* query 1 */
                (1, ClickSat::Skip),
                (2, ClickSat::Skip),
                (3, ClickSat::Medium),
                (3, ClickSat::High),
                (4, ClickSat::Miss),
                (5, ClickSat::Miss),
                (6, ClickSat::Miss),
                (7, ClickSat::Miss),
                (8, ClickSat::Miss),
                (9, ClickSat::Miss),
                /* query 2 */
                (1, ClickSat::Skip),
                (2, ClickSat::Skip),
                (3, ClickSat::Skip),
                (3, ClickSat::Skip),
                (4, ClickSat::Skip),
                (5, ClickSat::Skip),
                (6, ClickSat::Medium),
                (7, ClickSat::Medium),
                (8, ClickSat::High),
                (9, ClickSat::Miss),
                /* query 3 */
                (1, ClickSat::Skip),
                (2, ClickSat::Medium),
                (3, ClickSat::Medium),
                (3, ClickSat::Medium),
                (5, ClickSat::Skip),
                (4, ClickSat::High),
                (6, ClickSat::Miss),
                (7, ClickSat::Miss),
                (8, ClickSat::Miss),
                (9, ClickSat::Miss),
                /* query 4 */
                (1, ClickSat::Miss),
                (2, ClickSat::Miss),
                (3, ClickSat::Miss),
                (3, ClickSat::Miss),
                (5, ClickSat::Miss),
                (4, ClickSat::Miss),
                (6, ClickSat::Miss),
                (7, ClickSat::Miss),
                (8, ClickSat::Miss),
                (9, ClickSat::Miss),
            ],
        );

        let QueryFeatures {
            click_entropy,
            num_terms,
            mean_query_counter,
            mean_occurs_per_session,
            num_occurs,
            click_mrr,
            mean_clicks,
            mean_non_click,
        } = QueryFeatures::exact(
            &history,
            &Query {
                id: 423,
                words: vec![2, 100, 4],
            },
        );

        assert_approx_eq!(f32, click_entropy, 2.725_480_6);
        assert_eq!(num_terms, 3);

        assert_approx_eq!(f32, mean_query_counter, 1.5);
        assert_approx_eq!(f32, mean_occurs_per_session, 4.0);
        assert_eq!(num_occurs, 4);

        assert_approx_eq!(f32, click_mrr, 0.249_530_15);
        assert_approx_eq!(f32, mean_clicks, 2.25);
        assert_approx_eq!(f32, mean_non_click, 7.75);
    }
}
