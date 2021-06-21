use super::{click_entropy, HistSearchResult, Rank};
use crate::SessionId;
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
/// Click counter.
pub(super) struct ClickCounts {
    /// Click count of results ranked 1-2.
    pub(super) click12: u32,
    /// Click count of results ranked 3-5.
    pub(super) click345: u32,
    /// Click count of results ranked 6 upwards.
    pub(super) click6up: u32,
}

impl ClickCounts {
    fn new() -> Self {
        Self {
            click12: 0,
            click345: 0,
            click6up: 0,
        }
    }

    fn incr(mut self, rank: Rank) -> Self {
        match rank.0 {
            0 | 1 => self.click12 += 1,
            2 | 3 | 4 => self.click345 += 1,
            _ => self.click6up += 1,
        };
        self
    }
}

#[derive(Clone)]
/// Click habits and other features specific to the user.
pub(super) struct UserFeatures {
    /// Entropy over ranks of clicked results.
    pub(super) click_entropy: f32,
    /// Click counts of results ranked 1-2, 3-6, 6-10 resp.
    pub(super) click_counts: ClickCounts,
    /// Total number of search queries over all sessions.
    pub(super) num_queries: usize,
    /// Mean number of words per query.
    pub(super) mean_words_per_query: f32,
    /// Mean number of unique query words per session.
    pub(super) mean_unique_words_per_session: f32,
}

impl UserFeatures {
    /// Build user features for the given historical search results of the user.
    pub(super) fn build(hists: &[HistSearchResult]) -> Self {
        if hists.is_empty() {
            return Self {
                click_entropy: 0.,
                click_counts: ClickCounts::new(),
                num_queries: 0,
                mean_words_per_query: 0.,
                mean_unique_words_per_session: 0.,
            };
        }

        // query data for all search results over all sessions
        let all_results = hists.iter().map(|hist| &hist.query).collect::<HashSet<_>>();

        let click_entropy = click_entropy(hists);
        let click_counts = click_counts(hists);
        let num_queries = all_results.len();
        let words_per_query = all_results
            .iter()
            .map(|q| q.query_words.len())
            .sum::<usize>() as f32
            / num_queries as f32;

        let words_per_session = words_per_session(
            all_results
                .into_iter()
                .map(|q| (q.session_id, &q.query_words)),
        );

        Self {
            click_entropy,
            click_counts,
            num_queries,
            mean_words_per_query: words_per_query,
            mean_unique_words_per_session: words_per_session,
        }
    }
}

/// Calculate click counts of results ranked 1-2, 3-6, 6-10 resp.
fn click_counts(results: &[HistSearchResult]) -> ClickCounts {
    results
        .iter()
        .filter(|hist| hist.is_clicked())
        .fold(ClickCounts::new(), |clicks, hist| {
            clicks.incr(hist.final_rank)
        })
}

/// Calculate mean number of unique query words per session.
///
/// `results` is an iterator of `(session_id, query_words)` tuples over all results of the search history.
fn words_per_session<'a>(results: impl Iterator<Item = (SessionId, &'a Vec<String>)>) -> f32 {
    let words_by_session = results.fold(HashMap::new(), |mut words_by_session, (s, ws)| {
        let words = words_by_session.entry(s).or_insert_with(HashSet::new);
        words.extend(ws);
        words_by_session
    });

    let num_sessions = words_by_session.len();
    words_by_session
        .into_iter()
        .map(|(_, words)| words.len())
        .sum::<usize>() as f32
        / num_sessions as f32
}

#[cfg(test)]
mod tests {
    use crate::utils::mock_uuid;

    use super::{
        super::{Action, DayOfWeek, Query, QueryId, Rank, SessionId},
        *,
    };

    fn history<'a>(
        iter: impl IntoIterator<Item = &'a (usize, usize, &'a [&'a str], Action)>,
    ) -> Vec<HistSearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (session_id, query_id, query_words, action))| {
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                HistSearchResult {
                    query: Query {
                        session_id: SessionId(mock_uuid(*session_id)),
                        query_count: per_query_id,
                        query_id: QueryId(mock_uuid(*query_id)),
                        query_words: query_words.iter().map(|s| (**s).to_owned()).collect(),
                    },
                    url: in_query_id.to_string(),
                    domain: in_query_id.to_string(),
                    final_rank: Rank(in_query_id),
                    day: DayOfWeek::Tue,
                    action: *action,
                }
            })
            .collect()
    }

    #[test]
    fn the_right_statistics_are_computed() {
        let history = history(&[
            /* query 1 */
            (1, 2, &["23", "445"] as &[_], Action::Skip),
            (1, 2, &["23", "445"], Action::Skip),
            (1, 2, &["23", "445"], Action::Click1),
            (1, 2, &["23", "445"], Action::Click2),
            (1, 2, &["23", "445"], Action::Miss),
            (1, 2, &["23", "445"], Action::Miss),
            (1, 2, &["23", "445"], Action::Miss),
            (1, 2, &["23", "445"], Action::Miss),
            (1, 2, &["23", "445"], Action::Miss),
            (1, 2, &["23", "445"], Action::Miss),
            /* query 2 */
            (2, 33, &["48", "48", "48"], Action::Skip),
            (2, 33, &["48", "48", "48"], Action::Skip),
            (2, 33, &["48", "48", "48"], Action::Skip),
            (2, 33, &["48", "48", "48"], Action::Skip),
            (2, 33, &["48", "48", "48"], Action::Skip),
            (2, 33, &["48", "48", "48"], Action::Skip),
            (2, 33, &["48", "48", "48"], Action::Click1),
            (2, 33, &["48", "48", "48"], Action::Click1),
            (2, 33, &["48", "48", "48"], Action::Click2),
            (2, 33, &["48", "48", "48"], Action::Miss),
            /* query 3 */
            (1, 3, &["321", "12"], Action::Skip),
            (1, 3, &["321", "12"], Action::Click1),
            (1, 3, &["321", "12"], Action::Click1),
            (1, 3, &["321", "12"], Action::Click1),
            (1, 3, &["321", "12"], Action::Click2),
            (1, 3, &["321", "12"], Action::Miss),
            (1, 3, &["321", "12"], Action::Miss),
            (1, 3, &["321", "12"], Action::Miss),
            (1, 3, &["321", "12"], Action::Miss),
            (1, 3, &["321", "12"], Action::Miss),
        ]);

        let UserFeatures {
            click_entropy,
            click_counts,
            num_queries,
            mean_words_per_query,
            mean_unique_words_per_session,
        } = UserFeatures::build(&history);

        let ClickCounts {
            click12,
            click345,
            click6up,
        } = click_counts;

        assert_eq!(click12, 1);
        assert_eq!(click345, 5);
        assert_eq!(click6up, 3);

        assert_approx_eq!(f32, click_entropy, 2.725_480_6);

        assert_eq!(num_queries, 3);
        assert_approx_eq!(f32, mean_words_per_query, 2.333_333_3);
        assert_approx_eq!(f32, mean_unique_words_per_session, 2.5);
    }

    #[test]
    fn test_empty_user_history() {
        let UserFeatures {
            click_entropy,
            click_counts,
            num_queries,
            mean_words_per_query,
            mean_unique_words_per_session,
        } = UserFeatures::build(&[]);

        let ClickCounts {
            click12,
            click345,
            click6up,
        } = click_counts;

        assert_eq!(click12, 0);
        assert_eq!(click345, 0);
        assert_eq!(click6up, 0);

        assert_approx_eq!(f32, click_entropy, 0.0, ulps = 0);

        assert_eq!(num_queries, 0);
        assert_approx_eq!(f32, mean_words_per_query, 0.0, ulps = 0);
        assert_approx_eq!(f32, mean_unique_words_per_session, 0.0, ulps = 0);
    }
}
