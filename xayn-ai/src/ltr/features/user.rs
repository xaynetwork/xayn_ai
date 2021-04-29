#![allow(dead_code)] // TEMP

use crate::ltr::features::dataiku::{click_entropy, ClickSat, Rank, SearchResult};
use std::collections::{HashMap, HashSet};

/// Click counter.
struct ClickCounts {
    /// Click count of results ranked 1-2.
    click12: u32,
    /// Click count of results ranked 3-5.
    click345: u32,
    /// Click count of results ranked 6 upwards.
    click6up: u32,
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

struct UserFeatures {
    /// Entropy over ranks of clicked results.
    click_entropy: f32,
    /// Click counts of results ranked 1-2, 3-6, 6-10 resp.
    click_counts: ClickCounts,
    /// Total number of search queries over all sessions.
    num_queries: usize,
    /// Mean number of words per query.
    words_per_query: f32,
    /// Mean number of unique query words per session.
    words_per_session: f32,
}

/// Calculate user features for the given historical search results of a user.
fn user_features(history: &[SearchResult]) -> UserFeatures {
    let click_entropy = click_entropy(history);
    let click_counts = click_counts(history);

    // all queries over all sessions
    let all_queries = history
        .iter()
        .map(|r| (r.session_id, r.query_id, &r.query_words, r.query_counter))
        .collect::<HashSet<_>>();

    let num_queries = all_queries.len();
    let words_per_query = all_queries
        .iter()
        .map(|(_, _, words, _)| words.len())
        .sum::<usize>() as f32
        / num_queries as f32;

    let words_per_session = words_per_session(all_queries.into_iter());

    UserFeatures {
        click_entropy,
        click_counts,
        num_queries,
        words_per_query,
        words_per_session,
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
fn words_per_session<'a>(all_queries: impl Iterator<Item = (i32, i32, &'a Vec<i32>, u8)>) -> f32 {
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
