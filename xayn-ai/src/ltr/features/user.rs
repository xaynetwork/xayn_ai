#![allow(dead_code)] // TEMP

use crate::ltr::features::dataiku::{click_entropy, ClickSat, Rank, SearchResult};
use std::collections::{HashMap, HashSet};

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
        use Rank::*;
        match rank {
            First | Second => self.click12 += 1,
            Third | Fourth | Fifth => self.click345 += 1,
            _ => self.click6up += 1,
        };
        self
    }
}

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
    pub(super) fn extract(history: &[SearchResult]) -> UserFeatures {
        //FIXME temp. to make reviews easier by not showing the whole `user_features` function as changed
        user_features(history)
    }
}

/// Calculate user features for the given historical search results of a user.
fn user_features(history: &[SearchResult]) -> UserFeatures {
    if history.is_empty() {
        return UserFeatures {
            click_entropy: 0.0,
            click_counts: ClickCounts {
                click12: 0,
                click345: 0,
                click6up: 0,
            },
            num_queries: 0,
            mean_words_per_query: 0.0,
            mean_unique_words_per_session: 0.0,
        };
    }

    let click_entropy = click_entropy(history);
    let click_counts = click_counts(history);

    // FIXME soundgarden (and this code) takes the unique of `(session_id, query_id, words, query_counter)` tuples,
    //   but as far as I know there never should be:
    //   - same query_id different words
    //   - multiple equal (session_id, query_counter)
    //   So we should not need it?
    //   And on long histories it might be costly? Maybe?
    //
    // all search query results over all sessions
    let search_results = history
        .iter()
        .map(|r| (r.session_id, r.query_id, &*r.query_words, r.query_counter))
        .collect::<HashSet<_>>();

    let num_queries = search_results.len();
    let mean_words_per_query = search_results
        .iter()
        .map(|(_, _, words, _)| words.len())
        .sum::<usize>() as f32
        / num_queries as f32;

    let mean_unique_words_per_session = mean_unique_words_per_session(search_results.into_iter());

    UserFeatures {
        click_entropy,
        click_counts,
        num_queries,
        mean_words_per_query,
        mean_unique_words_per_session,
    }
}

/// Calculate click counts of results ranked 1-2, 3-6, 6-10 resp.
fn click_counts(results: &[SearchResult]) -> ClickCounts {
    results
        .iter()
        .filter(|r| r.relevance > ClickSat::Low)
        .fold(ClickCounts::new(), |counter, r| counter.incr(r.position))
}

/// Calculate mean number of unique query words per session.
fn mean_unique_words_per_session<'a>(
    all_queries: impl Iterator<Item = (i32, i32, &'a [i32], u8)>,
) -> f32 {
    let words_by_session =
        all_queries.fold(HashMap::new(), |mut words_by_session, (s, _, ws, _)| {
            let words = words_by_session
                .entry(s)
                .or_insert_with(HashSet::<i32>::new);
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
    use crate::ltr::features::dataiku::DayOfWeek;

    use super::*;

    fn history<'a>(
        iter: impl IntoIterator<Item = &'a (i32, i32, &'a [i32], ClickSat)>,
    ) -> Vec<SearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (session_id, query_id, query_words, relevance))| {
                let id = id as i32;
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                SearchResult {
                    session_id: *session_id,
                    user_id: 1,
                    query_id: *query_id,
                    day: DayOfWeek::Tue,
                    query_words: query_words.to_vec(),
                    url: in_query_id,
                    domain: in_query_id,
                    relevance: *relevance,
                    position: Rank::from_usize(in_query_id as usize),
                    query_counter: per_query_id as u8,
                }
            })
            .collect()
    }

    #[test]
    fn the_right_statistics_are_computed() {
        let history = history(&[
            /* query 1 */
            (1, 2, &[23, 445] as &[_], ClickSat::Skip),
            (1, 2, &[23, 445], ClickSat::Skip),
            (1, 2, &[23, 445], ClickSat::Medium),
            (1, 2, &[23, 445], ClickSat::High),
            (1, 2, &[23, 445], ClickSat::Miss),
            (1, 2, &[23, 445], ClickSat::Miss),
            (1, 2, &[23, 445], ClickSat::Miss),
            (1, 2, &[23, 445], ClickSat::Miss),
            (1, 2, &[23, 445], ClickSat::Miss),
            (1, 2, &[23, 445], ClickSat::Miss),
            /* query 2 */
            (2, 33, &[48, 48, 48], ClickSat::Skip),
            (2, 33, &[48, 48, 48], ClickSat::Skip),
            (2, 33, &[48, 48, 48], ClickSat::Skip),
            (2, 33, &[48, 48, 48], ClickSat::Skip),
            (2, 33, &[48, 48, 48], ClickSat::Skip),
            (2, 33, &[48, 48, 48], ClickSat::Skip),
            (2, 33, &[48, 48, 48], ClickSat::Medium),
            (2, 33, &[48, 48, 48], ClickSat::Medium),
            (2, 33, &[48, 48, 48], ClickSat::High),
            (2, 33, &[48, 48, 48], ClickSat::Miss),
            /* query 3 */
            (1, 3, &[321, 12], ClickSat::Skip),
            (1, 3, &[321, 12], ClickSat::Medium),
            (1, 3, &[321, 12], ClickSat::Medium),
            (1, 3, &[321, 12], ClickSat::Medium),
            (1, 3, &[321, 12], ClickSat::High),
            (1, 3, &[321, 12], ClickSat::Miss),
            (1, 3, &[321, 12], ClickSat::Miss),
            (1, 3, &[321, 12], ClickSat::Miss),
            (1, 3, &[321, 12], ClickSat::Miss),
            (1, 3, &[321, 12], ClickSat::Miss),
        ]);

        let UserFeatures {
            click_entropy,
            click_counts,
            num_queries,
            mean_words_per_query,
            mean_unique_words_per_session,
        } = UserFeatures::extract(&history);

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
        } = UserFeatures::extract(&[]);

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
