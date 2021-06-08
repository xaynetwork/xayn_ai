#![allow(dead_code)] // TEMP

use super::{click_entropy, Action, HistSearchResult, Rank};
use crate::SessionId;
use std::collections::{HashMap, HashSet};

/// Click counter.
pub(crate) struct ClickCounts {
    /// Click count of results ranked 1-2.
    pub(crate) click12: u32,
    /// Click count of results ranked 3-5.
    pub(crate) click345: u32,
    /// Click count of results ranked 6 upwards.
    pub(crate) click6up: u32,
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

/// Click habits and other features specific to the user.
pub(crate) struct UserFeatures {
    /// Entropy over ranks of clicked results.
    pub(crate) click_entropy: f32,
    /// Click counts of results ranked 1-2, 3-6, 6-10 resp.
    pub(crate) click_counts: ClickCounts,
    /// Total number of search queries over all sessions.
    pub(crate) num_queries: usize,
    /// Mean number of words per query.
    pub(crate) words_per_query: f32,
    /// Mean number of unique query words per session.
    pub(crate) words_per_session: f32,
}

impl UserFeatures {
    /// Build user features for the given historical search results of the user.
    pub(crate) fn build(history: &[HistSearchResult]) -> Self {
        if history.is_empty() {
            return Self {
                click_entropy: 0.,
                click_counts: ClickCounts::new(),
                num_queries: 0,
                words_per_query: 0.,
                words_per_session: 0.,
            };
        }

        // query data for all search results over all sessions
        let all_results = history.iter().map(|r| &r.query).collect::<HashSet<_>>();

        let click_entropy = click_entropy(history);
        let click_counts = click_counts(history);
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
            words_per_query,
            words_per_session,
        }
    }
}

/// Calculate click counts of results ranked 1-2, 3-6, 6-10 resp.
fn click_counts(results: &[HistSearchResult]) -> ClickCounts {
    results
        .iter()
        .filter(|r| r.action > Action::Click0)
        .fold(ClickCounts::new(), |counter, r| counter.incr(r.rerank))
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
