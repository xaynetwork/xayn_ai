#![allow(dead_code)] // TEMP

use itertools::Itertools;
use std::collections::{HashMap, HashSet};

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

struct QueryFeatures {
    click_entropy: f32,
    num_terms: usize,
    /// Average `n` where query is the `n`th of a session.
    rank_per_session: f32,
    /// Average number of occurrences per session.
    occurs_per_session: f32,
    /// Total number of occurrences.
    num_occurs: usize,
    click_mrr: f32,
    /// Average number of clicks.
    avg_clicks: f32,
    /// Average number of skips.
    avg_skips: f32,
}

fn query_features(history: &[SearchResult], query: Query) -> QueryFeatures {
    let history_q = history
        .iter()
        .filter(|r| r.query_id == query.id)
        .collect_vec();

    let click_entropy = click_entropy(&history_q);

    let occurs = history_q
        .iter()
        .map(|r| (r.session_id, r.query_counter))
        .collect::<HashSet<_>>();

    let num_occurs = occurs.len();

    let rank_sum = occurs
        .into_iter()
        .map(|(_, query_count)| query_count)
        .sum::<u8>() as f32;
    let rank_per_session = rank_sum / num_occurs as f32;

    let num_sessions = history_q.iter().unique_by(|r| r.session_id).count() as f32;
    let occurs_per_session = num_occurs as f32 / num_sessions;

    let clicked = history_q
        .iter()
        .filter(|r| r.relevance > ClickScore::Low)
        .collect_vec();
    let click_mrr = mrr(&clicked);

    let avg_clicks = clicked.len() as f32 / num_occurs as f32;

    let avg_skips = history_q
        .into_iter()
        .filter(|r| r.relevance == ClickScore::Low)
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

/// Mean reciprocal rank of the given results.
fn mrr(results: &[impl AsRef<SearchResult>]) -> f32 {
    let denom = 1. + results.len() as f32;
    let numer = 0.283
        + results
            .into_iter()
            .map(|r| f32::from(r.as_ref().position).recip())
            .sum::<f32>();

    numer / denom
}

struct Query {
    id: i32,
    words: Vec<i32>,
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

impl AsRef<SearchResult> for SearchResult {
    fn as_ref(&self) -> &SearchResult {
        self
    }
}

/// Yandex notion of dwell-time: time elapsed between a click and the next action.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
enum ClickScore {
    /// Less than 50 units of time.
    Low = 0,
    /// From 50 to 300 units of time.
    Medium,
    /// More than 300 units of time, or last click of the session.
    High,
}

enum UrlFeedback {
    /// Snippet examined and URL clicked.
    Click(ClickScore),
    /// Snippet examined but URL not clicked.
    Skip,
    /// Snippet not examined.
    Miss,
}

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
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

impl From<Rank> for f32 {
    fn from(rank: Rank) -> Self {
        rank as u8 as f32
    }
}

/// Counts the variety of query terms over a given test session.
fn terms_variety(query: &[SearchResult], session_id: i32) -> usize {
    query
        .iter()
        .filter(|r| r.session_id == session_id)
        .flat_map(|r| &r.query_words)
        .unique()
        .count()
}

/// Weekend seasonality of a given domain.
fn seasonality(history: &[SearchResult], domain: i32) -> f32 {
    let (clicks_wknd, clicks_wkday) = history
        .iter()
        .filter(|r| r.domain == domain && r.relevance > ClickScore::Low)
        .fold((0, 0), |(wknd, wkday), r| {
            // assume day 1 is Tue
            if r.day % 7 == 5 || r.day % 7 == 6 {
                (wknd + 1, wkday)
            } else {
                (wknd, wkday + 1)
            }
        });

    2.5 * (1. + clicks_wknd as f32) / (1. + clicks_wkday as f32)
}

fn prior(outcome: UrlFeedback) -> u8 {
    match outcome {
        UrlFeedback::Miss => 1,
        _ => 0,
    }
}

fn click_entropy(results: &[impl AsRef<SearchResult>]) -> f32 {
    let rank_freqs = results
        .iter()
        .filter_map(|r| (r.as_ref().relevance > ClickScore::Low).then(|| r.as_ref().position))
        .counts();

    let freqs_sum = rank_freqs.values().sum::<usize>() as f32;
    rank_freqs
        .into_iter()
        .map(|(_, freq)| {
            let prob = freq as f32 / freqs_sum;
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

    fn incr(mut self, rank: Rank) -> Self {
        use Rank::*;
        match rank {
            First | Second => self.click12 += 1,
            Third | Fourth | Fifth => self.click345 += 1,
            _ => self.click6to10 += 1,
        };
        self
    }
}

fn click_counts(results: &[SearchResult]) -> ClickCounts {
    results
        .iter()
        .filter(|r| r.relevance > ClickScore::Low)
        .fold(ClickCounts::new(), |counter, r| counter.incr(r.position))
}

fn user_features(history: &[SearchResult]) -> UserFeatures {
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

    let words_by_session =
        all_queries
            .into_iter()
            .fold(HashMap::new(), |mut words_by_session, (s, _, ws, _)| {
                let words = words_by_session
                    .entry(s)
                    .or_insert_with(HashSet::<i32>::new);
                words.extend(ws);
                words_by_session
            });

    let num_sessions = words_by_session.len();
    let words_per_session = words_by_session
        .into_iter()
        .map(|(_, words)| words.len())
        .sum::<usize>() as f32
        / num_sessions as f32;

    UserFeatures {
        click_entropy: click_entropy(history),
        click_counts: click_counts(history),
        num_queries,
        words_per_query,
        words_per_session,
    }
}
