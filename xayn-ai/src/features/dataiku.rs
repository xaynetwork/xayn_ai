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
        .map(|(_, query_count)| query_count as f32)
        .sum::<f32>();
    let rank_per_session = rank_sum / num_occurs as f32;

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

/// Mean reciprocal rank of results filtered by outcome and a predicate.
fn mean_recip_rank(
    results: &[impl AsRef<SearchResult>],
    outcome: Option<MrrOutcome>,
    pred: Option<FilterPred>,
) -> f32 {
    let filtered = results
        .iter()
        .filter(|r| {
            let relevance = r.as_ref().relevance;
            match outcome {
                Some(MrrOutcome::Miss) => relevance == ClickSat::Miss,
                Some(MrrOutcome::Skip) => relevance == ClickSat::Skip,
                Some(MrrOutcome::Click) => relevance > ClickSat::Low,
                None => true,
            }
        })
        .filter(|r| pred.map_or(true, |p| p.apply(r)))
        .collect_vec();

    let denom = 1. + filtered.len() as f32;
    let numer = 0.283 // prior recip rank assuming uniform distributed ranks
        + filtered
            .into_iter()
            .map(|r| f32::from(r.as_ref().position).recip())
            .sum::<f32>();

    numer / denom
}

struct Query {
    id: i32,
    words: Vec<i32>,
}

struct SearchResult {
    session_id: i32,
    user_id: i32,
    query_id: i32,
    day: u8,
    query_words: Vec<i32>,
    url: i32,
    domain: i32,
    relevance: ClickSat,
    position: Rank,
    query_counter: u8,
}

impl AsRef<SearchResult> for SearchResult {
    fn as_ref(&self) -> &SearchResult {
        self
    }
}

/// Click satisfaction score.
///
/// Based on Yandex notion of dwell-time: time elapsed between a click and the next action.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
enum ClickSat {
    /// Snippet examined but URL not clicked.
    Skip,
    /// Snippet not examined.
    Miss,
    /// Less than 50 units of time or no click.
    Low,
    /// From 50 to 300 units of time.
    Medium,
    /// More than 300 units of time or last click of the session.
    High,
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
        .filter(|r| r.domain == domain && r.relevance > ClickSat::Low)
        .fold((0, 0), |(wknd, wkday), r| {
            // dataiku observation of Yandex dataset: day 1 is Tue
            if r.day % 7 == 5 || r.day % 7 == 6 {
                (wknd + 1, wkday)
            } else {
                (wknd, wkday + 1)
            }
        });

    2.5 * (1. + clicks_wknd as f32) / (1. + clicks_wkday as f32)
}

/// Entropy over the rank of the given results that were clicked.
fn click_entropy(results: &[impl AsRef<SearchResult>]) -> f32 {
    let rank_freqs = results
        .iter()
        .filter_map(|r| {
            let r = r.as_ref();
            (r.relevance > ClickSat::Low).then(|| r.position)
        })
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

/// Click counter.
struct ClickCounts {
    /// Click count of results ranked 1-2.
    click12: u32,
    /// Click count of results ranked 3-5.
    click345: u32,
    /// Click count of results ranked 6-10.
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
        .filter(|r| r.relevance > ClickSat::Low)
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MrrOutcome {
    Miss,
    Skip,
    Click,
}

/// Atomic features of which an aggregate feature is composed of.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum AtomFeat {
    /// MRR for miss, skip, click.
    MeanRecipRank(MrrOutcome),
    /// MRR for all outcomes.
    MeanRecipRankAll,
    /// Conditional probabilities for miss, skip, click0, click1, click2.
    CondProb(ClickSat),
    /// Snippet quality.
    SnippetQuality,
}

type FeatMap = HashMap<AtomFeat, f32>;

/// Cumulated features for a given user.
struct CumFeatures {
    /// Cumulated feature for matching URL.
    url: FeatMap,
}

fn cum_features(hist: &[SearchResult], res: SearchResult) -> CumFeatures {
    let url = hist
        .iter()
        // if res is ranked n, get the n-1 results ranked above res
        .filter(|r| {
            r.session_id == res.session_id
                && r.query_id == res.query_id
                && r.query_counter == res.query_counter
                && r.position < res.position
        })
        // calculate specified cond probs for each of the above
        .flat_map(|r| {
            let pred = FilterPred::new(UrlOrDom::Url(r.url));
            pred.cum_spec()
                .into_iter()
                .map(move |outcome| (outcome, cond_prob(hist, outcome, pred)))
        })
        // sum cond probs for each outcome
        .fold(HashMap::new(), |mut cp_map, (outcome, cp)| {
            let sum = cp_map.entry(AtomFeat::CondProb(outcome)).or_insert(0.);
            *sum += cp;
            cp_map
        });

    CumFeatures { url }
}

/// Aggregate features for a given user.
struct AggregFeatures {
    /// Aggregate feature for matching domain.
    dom: FeatMap,
    /// Aggregate feature for matching domain over anterior sessions.
    dom_ant: FeatMap,
    /// Aggregate feature for matching URL.
    url: FeatMap,
    /// Aggregate feature for matching URL over anterior sessions.
    url_ant: FeatMap,
    /// Aggregate feature for matching domain and query.
    dom_query: FeatMap,
    /// Aggregate feature for matching domain and query over anterior sessions.
    dom_query_ant: FeatMap,
    /// Aggregate feature for matching URL and query.
    url_query: FeatMap,
    /// Aggregate feature for matching URL and query over anterior sessions.
    url_query_ant: FeatMap,
    /// Aggregate feature for matching URL and query over current session.
    url_query_curr: FeatMap,
}

fn aggreg_features(hist: &[SearchResult], doc: DocAddr, query: Query, sess: i32) -> AggregFeatures {
    let anterior = SessionCond::Anterior(sess);
    let current = SessionCond::Current(sess);

    let pred_dom = FilterPred::new(doc.dom);
    let dom = aggreg_feat(hist, pred_dom);
    let dom_ant = aggreg_feat(hist, pred_dom.with_session(anterior));

    let pred_url = FilterPred::new(doc.url);
    let url = aggreg_feat(hist, pred_url);
    let url_ant = aggreg_feat(hist, pred_url.with_session(anterior));

    let pred_dom_query = pred_dom.with_query(query.id);
    let dom_query = aggreg_feat(hist, pred_dom_query);
    let dom_query_ant = aggreg_feat(hist, pred_dom_query.with_session(anterior));

    let pred_url_query = pred_url.with_query(query.id);
    let url_query = aggreg_feat(hist, pred_url_query);
    let url_query_ant = aggreg_feat(hist, pred_url_query.with_session(anterior));
    let url_query_curr = aggreg_feat(hist, pred_url_query.with_session(current));

    AggregFeatures {
        dom,
        dom_ant,
        url,
        url_ant,
        dom_query,
        dom_query_ant,
        url_query,
        url_query_ant,
        url_query_curr,
    }
}

fn aggreg_feat(hist: &[SearchResult], pred: FilterPred) -> FeatMap {
    let eval_atom = |atom_feat| match atom_feat {
        AtomFeat::MeanRecipRank(outcome) => mean_recip_rank(hist, Some(outcome), Some(pred)),
        AtomFeat::MeanRecipRankAll => mean_recip_rank(hist, None, Some(pred)),
        AtomFeat::SnippetQuality => snippet_quality(),
        AtomFeat::CondProb(outcome) => cond_prob(hist, outcome, pred),
    };
    pred.agg_spec()
        .into_iter()
        .map(|atom_feat| (atom_feat, eval_atom(atom_feat)))
        .collect()
}

/// Quality of the snippet associated with a search result.
fn snippet_quality() -> f32 {
    todo!()
}

/// Probability of an outcome conditioned on some predicate.
fn cond_prob(hist: &[SearchResult], outcome: ClickSat, pred: FilterPred) -> f32 {
    let prior = if outcome == ClickSat::Miss { 1 } else { 0 };

    let filtered_by_pred = hist.iter().filter(|r| pred.apply(r)).collect_vec();
    let denom = 1 + filtered_by_pred.len();
    let numer = prior
        + filtered_by_pred
            .into_iter()
            .filter(|r| r.relevance == outcome)
            .count();

    numer as f32 / denom as f32
}

struct DocAddr {
    url: UrlOrDom,
    dom: UrlOrDom,
}

impl DocAddr {
    fn new(url: i32, dom: i32) -> Self {
        Self {
            url: UrlOrDom::Url(url),
            dom: UrlOrDom::Dom(dom),
        }
    }
}

#[derive(Clone, Copy)]
enum UrlOrDom {
    /// A specific URL.
    Url(i32),
    /// Any URL belonging to the given domain.
    Dom(i32),
}

/// Query submission timescale.
#[derive(Clone, Copy)]
enum SessionCond {
    /// Before current session.
    Anterior(i32),
    /// Current session.
    Current(i32),
    /// All historic.
    All,
}

/// Filter predicate representing a boolean condition on a search result.
#[derive(Clone, Copy)]
struct FilterPred {
    doc: UrlOrDom,
    query: Option<i32>,
    session: SessionCond,
}

impl FilterPred {
    fn new(doc: UrlOrDom) -> Self {
        Self {
            doc,
            query: None,
            session: SessionCond::All,
        }
    }

    fn with_query(mut self, query_id: i32) -> Self {
        self.query = Some(query_id);
        self
    }

    fn with_session(mut self, session: SessionCond) -> Self {
        self.session = session;
        self
    }

    /// Lookup the specification of the aggregate feature for this filter predicate.
    fn agg_spec(&self) -> Vec<AtomFeat> {
        use AtomFeat::{
            CondProb as CP,
            MeanRecipRank as MRR,
            MeanRecipRankAll as mrr,
            SnippetQuality as SQ,
        };
        use ClickSat::{High as click2, Miss as miss, Skip as skip};
        use MrrOutcome::*;
        use SessionCond::{All, Anterior as Ant, Current};
        use UrlOrDom::*;

        match (self.doc, self.query, self.session) {
            (Dom(_), None, All) => vec![CP(skip), CP(miss), CP(click2), SQ],
            (Dom(_), None, Ant(_)) => vec![CP(click2), CP(miss), SQ],
            (Url(_), None, All) => vec![MRR(Click), CP(click2), CP(miss), SQ],
            (Url(_), None, Ant(_)) => vec![CP(click2), CP(miss), SQ],
            (Dom(_), Some(_), All) => vec![CP(miss), SQ, MRR(Miss)],
            (Dom(_), Some(_), Ant(_)) => vec![SQ],
            (Url(_), Some(_), All) => vec![mrr, CP(click2), CP(miss), SQ],
            (Url(_), Some(_), Ant(_)) => vec![mrr, MRR(Click), MRR(Miss), MRR(Skip), CP(skip), SQ],
            (Url(_), Some(_), Current(_)) => vec![MRR(Miss)],
            _ => vec![],
        }
    }

    /// Lookup the specification of the cumulated feature for this filter predicate.
    fn cum_spec(&self) -> Vec<ClickSat> {
        use ClickSat::{High as click2, Medium as click1, Skip as skip};
        use SessionCond::All;
        use UrlOrDom::*;

        match (self.doc, self.query, self.session) {
            (Url(_), None, All) => vec![skip, click1, click2],
            _ => vec![],
        }
    }

    /// Applies the predicate to the given search result.
    fn apply(&self, r: impl AsRef<SearchResult>) -> bool {
        let r = r.as_ref();
        let doc_cond = match self.doc {
            UrlOrDom::Url(url) => r.url == url,
            UrlOrDom::Dom(dom) => r.domain == dom,
        };
        let query_cond = match self.query {
            Some(id) => r.query_id == id,
            None => true,
        };
        let session_id = r.session_id;
        let session_cond = match self.session {
            SessionCond::Anterior(id) => session_id < id,
            SessionCond::Current(id) => session_id == id,
            SessionCond::All => true,
        };
        doc_cond && query_cond && session_cond
    }
}
