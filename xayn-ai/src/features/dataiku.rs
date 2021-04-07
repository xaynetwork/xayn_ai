#![allow(dead_code)] // TEMP

use std::collections::{HashMap, HashSet};

struct UserFeatures {
    /// entropy over ranks of clicked results
    click_entropy: f32,
    /// click counts of results ranked 1-2, 3-6, 6-10 resp.
    click_counts: ClickCounts,
    /// total number of search queries over all sessions
    num_queries: usize,
    /// mean number of terms per query
    num_query_terms: f32,
    /// mean number of unique query terms per session
    num_query_terms_session: f32,
}

struct QueryFeatures {
    click_entropy: f32,
    num_terms: u32,
    /// average rank per session
    rank_session: u32,
    /// average number of occurrences per session
    occurrences_session: u32,
    /// total number of occurrences
    occurrences: u32,
    click_mrr: f32,
    /// average number of clicks
    num_clicks: u32,
    /// average number of skips
    num_skips: u32,
}

#[derive(Clone)]
struct SearchResult {
    session_id: i32,
    user_id: i32,
    query_id: i32,
    day: u8,
    query_words: Vec<i32>,
    url: i32,
    domain: i32,
    relevance: ClickScore,
    position: Rank,
    query_counter: u8,
}

// Yandex dwell-time
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
enum ClickScore {
    Low = 0, // < 50 units of time
    Medium,  // 50-300 units of time
    High,    // > 300 units of time
}

enum UrlFeedback {
    Click(ClickScore), // snippet examined, clicked
    Skip,              // examined but not clicked
    Miss,              // not examined
}

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
enum Rank {
    First = 1,
    Second,
    Third,
    Fourth,
    Fifth,
    Sixth,
    Seventh,
    Eighth,
    Nineth,
    Tenth,
}

fn query_features(_history: &[SearchResult], _query: &[SearchResult]) -> QueryFeatures {
    unimplemented!();
}

fn seasonality(_history: &[SearchResult], _domain: i32) -> f32 {
    0.
}

fn prior(outcome: UrlFeedback) -> u8 {
    match outcome {
        UrlFeedback::Miss => 1,
        _ => 0,
    }
}

fn click_entropy(results: &[SearchResult]) -> f32 {
    let (rank_freqs, count) = results
        .iter()
        .filter(|r| r.relevance > ClickScore::Low)
        .cloned()
        .map(|r| r.position)
        .fold((HashMap::new(), 0), |(mut freqs, count), rank| {
            let freq = freqs.entry(rank).or_insert(0);
            *freq += 1;
            (freqs, count + 1)
        });

    rank_freqs
        .into_iter()
        .map(|(_, freq)| {
            let prob = freq as f32 / count as f32;
            -prob * prob.log2()
        })
        .sum()
}

struct ClickCounts {
    click12: u32,
    click345: u32,
    click6to10: u32,
}

impl ClickCounts {
    fn new() -> Self {
        Self {
            click12: 0,
            click345: 0,
            click6to10: 0,
        }
    }

    fn inc_12(mut self) -> Self {
        self.click12 += 1;
        self
    }

    fn inc_345(mut self) -> Self {
        self.click345 += 1;
        self
    }

    fn inc_6to10(mut self) -> Self {
        self.click6to10 += 1;
        self
    }
}

fn click_counts(results: &[SearchResult]) -> ClickCounts {
    results
        .iter()
        .filter(|r| r.relevance > ClickScore::Low)
        .fold(ClickCounts::new(), |counter, r| {
            use Rank::*;
            match r.position {
                First | Second => counter.inc_12(),
                Third | Fourth | Fifth => counter.inc_345(),
                _ => counter.inc_6to10(),
            }
        })
}

fn user_features(history: &[SearchResult]) -> UserFeatures {
    let click_entropy = click_entropy(history);
    let click_counts = click_counts(history);

    let all_queries = history
        .iter()
        .cloned()
        .map(|r| (r.session_id, r.query_id, r.query_words, r.query_counter))
        .collect::<HashSet<_>>();

    let num_queries = all_queries.len();
    let num_query_terms = all_queries
        .iter()
        .map(|(_, _, words, _)| words.len())
        .sum::<usize>() as f32
        / num_queries as f32;

    let words_by_session =
        all_queries
            .into_iter()
            .fold(HashMap::new(), |mut words_by_session, (s, _, ws, _)| {
                let words = words_by_session.entry(s).or_insert(HashSet::<i32>::new());
                words.extend(ws);
                words_by_session
            });

    let num_sessions = words_by_session.len();
    let num_query_terms_session = words_by_session
        .into_iter()
        .map(|(_, words)| words.len())
        .sum::<usize>() as f32
        / num_sessions as f32;

    UserFeatures {
        click_entropy,
        click_counts,
        num_queries,
        num_query_terms,
        num_query_terms_session,
    }
}
