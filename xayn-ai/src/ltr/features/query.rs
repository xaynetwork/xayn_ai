use super::{click_entropy, mean_recip_rank, HistSearchResult, Query};
use itertools::Itertools;
use std::collections::HashSet;

#[derive(Clone)]
/// Features specific to a given query.
pub(super) struct QueryFeatures {
    /// Entropy over ranks of clicked results.
    pub(super) click_entropy: f32,
    /// Number of terms.
    pub(super) num_terms: usize,
    /// Average `n` where query is the `n`th of a session.
    pub(super) mean_query_count: f32,
    /// Average number of occurrences per session.
    pub(super) mean_occurs_per_session: f32,
    /// Total number of occurrences.
    pub(super) num_occurs: usize,
    /// Mean reciprocal rank of clicked results.
    pub(super) click_mrr: f32,
    /// Average number of clicks.
    pub(super) mean_clicks: f32,
    /// Average number of non-clicks.
    ///
    /// The python reference implementation implies this is supposed to be the
    /// mean number of skips. But it is implemented as the mean number of non
    /// clicks.
    pub(super) mean_non_clicks: f32,
}

impl QueryFeatures {
    /// Build query features for the given query and history of the user.
    pub(super) fn build(hists: &[HistSearchResult], query: &Query) -> Self {
        // history filtered by query
        let hists_q = hists
            .iter()
            .filter(|hist| hist.query.query_id == query.query_id)
            .collect_vec();

        let num_terms = query.query_words.len();

        if hists_q.is_empty() {
            return Self {
                click_entropy: 0.,
                num_terms,
                mean_query_count: 0.,
                mean_occurs_per_session: 0.,
                num_occurs: 0,
                click_mrr: 0.283,
                mean_clicks: 0.,
                mean_non_clicks: 0.,
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

        let mean_non_clicks = hists_q
            .into_iter()
            .filter(|hist| !hist.is_clicked())
            .count() as f32
            / num_occurs as f32;

        Self {
            click_entropy,
            num_terms,
            mean_query_count,
            mean_occurs_per_session: occurs_per_session,
            num_occurs,
            click_mrr,
            mean_clicks,
            mean_non_clicks,
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

#[cfg(test)]
mod tests {
    use super::{
        super::{Action, DayOfWeek, QueryId, Rank, SessionId},
        *,
    };
    use crate::tests::mock_uuid;
    use test_utils::assert_approx_eq;

    #[test]
    fn test_query_features_no_matching_history() {
        let QueryFeatures {
            click_entropy,
            num_terms,
            mean_query_count,
            mean_occurs_per_session,
            num_occurs,
            click_mrr,
            mean_clicks,
            mean_non_clicks,
        } = QueryFeatures::build(
            &[],
            &Query {
                session_id: SessionId(mock_uuid(10)),
                query_count: 1,
                query_id: QueryId(mock_uuid(233)),
                query_words: vec!["2".to_owned(), "100".to_owned(), "4".to_owned()],
            },
        );

        assert_approx_eq!(f32, click_entropy, 0.0);
        assert_eq!(num_terms, 3);

        assert_approx_eq!(f32, mean_query_count, 0.0);
        assert_approx_eq!(f32, mean_occurs_per_session, 0.0);
        assert_eq!(num_occurs, 0);

        assert_approx_eq!(f32, click_mrr, 0.283);
        assert_approx_eq!(f32, mean_clicks, 0.0);
        assert_approx_eq!(f32, mean_non_clicks, 0.0);
    }

    fn history<'a>(
        query_id: QueryId,
        iter: impl IntoIterator<Item = &'a (&'a str, Action)>,
    ) -> Vec<HistSearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (domain, action))| {
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                HistSearchResult {
                    query: Query {
                        session_id: SessionId(mock_uuid(1)),
                        query_count: per_query_id,
                        query_id,
                        query_words: vec!["1".to_owned()],
                    },
                    url: in_query_id.to_string(),
                    domain: domain.to_string(),
                    final_rank: Rank(in_query_id),
                    day: DayOfWeek::Tue,
                    action: *action,
                }
            })
            .collect()
    }

    #[test]
    fn test_query_features() {
        let query_id = QueryId(mock_uuid(233));
        let history = history(
            query_id,
            &[
                /* query 1 */
                ("1", Action::Skip),
                ("2", Action::Skip),
                ("3", Action::Click1),
                ("3", Action::Click2),
                ("4", Action::Miss),
                ("5", Action::Miss),
                ("6", Action::Miss),
                ("7", Action::Miss),
                ("8", Action::Miss),
                ("9", Action::Miss),
                /* query 2 */
                ("1", Action::Skip),
                ("2", Action::Skip),
                ("3", Action::Skip),
                ("3", Action::Skip),
                ("4", Action::Skip),
                ("5", Action::Skip),
                ("6", Action::Click1),
                ("7", Action::Click1),
                ("8", Action::Click2),
                ("9", Action::Miss),
                /* query 3 */
                ("1", Action::Skip),
                ("2", Action::Click1),
                ("3", Action::Click1),
                ("3", Action::Click1),
                ("5", Action::Skip),
                ("4", Action::Click2),
                ("6", Action::Miss),
                ("7", Action::Miss),
                ("8", Action::Miss),
                ("9", Action::Miss),
                /* query 4 */
                ("1", Action::Miss),
                ("2", Action::Miss),
                ("3", Action::Miss),
                ("3", Action::Miss),
                ("5", Action::Miss),
                ("4", Action::Miss),
                ("6", Action::Miss),
                ("7", Action::Miss),
                ("8", Action::Miss),
                ("9", Action::Miss),
            ],
        );

        let QueryFeatures {
            click_entropy,
            num_terms,
            mean_query_count,
            mean_occurs_per_session,
            num_occurs,
            click_mrr,
            mean_clicks,
            mean_non_clicks,
        } = QueryFeatures::build(
            &history,
            &Query {
                session_id: SessionId(mock_uuid(10)),
                query_count: 1,
                query_id,
                query_words: vec!["2".to_owned(), "100".to_owned(), "4".to_owned()],
            },
        );

        assert_approx_eq!(f32, click_entropy, 2.725_480_6);
        assert_eq!(num_terms, 3);

        assert_approx_eq!(f32, mean_query_count, 1.5);
        assert_approx_eq!(f32, mean_occurs_per_session, 4.0);
        assert_eq!(num_occurs, 4);

        assert_approx_eq!(f32, click_mrr, 0.249_530_15);
        assert_approx_eq!(f32, mean_clicks, 2.25);
        assert_approx_eq!(f32, mean_non_clicks, 7.75);
    }
}
