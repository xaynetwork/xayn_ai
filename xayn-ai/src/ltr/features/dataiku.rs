//! Feature extraction algorithm based on Dataiku's solution to Yandex Personalized Web Search
//! Challenge [1]. See [2] for the first implementation in Python.
//!
//! [1]: https://www.academia.edu/9872579/Dataikus_Solution_to_Yandexs_Personalized_Web_Search_Challenge
//! [2]: https://github.com/xaynetwork/soundgarden

#![allow(dead_code)] // TEMP

use itertools::Itertools;
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;

use super::{
    aggreg_features,
    cum_features,
    query_features,
    user_features,
    AggregFeatures,
    CumFeatures,
    QueryFeatures,
    UserFeatures,
};

/// Click satisfaction score.
///
/// Based on Yandex notion of dwell-time: time elapsed between a click and the next action.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub(crate) enum ClickSat {
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
pub(crate) enum Rank {
    First = 1,
    Second,
    Third,
    Fourth,
    Fifth,
    Sixth,
    Seventh,
    Eighth,
    Nineth,
    Last,
}

impl From<Rank> for f32 {
    fn from(rank: Rank) -> Self {
        rank as u8 as f32
    }
}

#[derive(PartialEq)]
pub(crate) enum DayOfWeek {
    Mon,
    Tue,
    Wed,
    Thu,
    Fri,
    Sat,
    Sun,
}

/// Data pertaining to a single result from a search.
pub struct SearchResult {
    /// Session identifier.
    pub(crate) session_id: i32,
    /// User identifier.
    pub(crate) user_id: i32,
    /// Query identifier.
    pub(crate) query_id: i32,
    /// Day of week search was performed.
    pub(crate) day: DayOfWeek,
    /// Words of the query.
    pub(crate) query_words: Vec<String>,
    /// URL of result.
    pub(crate) url: String,
    /// Domain of result.
    pub(crate) domain: String,
    /// Relevance level of the result.
    pub(crate) relevance: ClickSat,
    /// Position among other results.
    pub(crate) position: Rank,
    /// Query count within session.
    pub(crate) query_counter: u8,
}

impl AsRef<SearchResult> for SearchResult {
    fn as_ref(&self) -> &SearchResult {
        self
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MrrOutcome {
    Miss,
    Skip,
    Click,
}

/// Atomic features of which an aggregate feature is composed of.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AtomFeat {
    /// MRR for miss, skip, click.
    MeanRecipRank(MrrOutcome),
    /// MRR for all outcomes.
    MeanRecipRankAll,
    /// Conditional probabilities for miss, skip, click0, click1, click2.
    CondProb(ClickSat),
    /// Snippet quality.
    SnippetQuality,
}

pub(crate) type FeatMap = HashMap<AtomFeat, f32>;

#[derive(Clone)]
/// Search results from some query.
struct ResultSet<'a>(Vec<&'a SearchResult>);

impl<'a> ResultSet<'a> {
    /// New result set.
    ///
    /// Assumes `rs[i]` contains the search result `r` with `r.position` `i+1`.
    // TODO may relax this assumption later
    fn new(rs: Vec<&'a SearchResult>) -> Self {
        Self(rs)
    }

    /// Search result at position `pos` of the result set.
    ///
    /// # Panic
    /// Panics if the length of the underlying vector is less than `pos`.
    fn get(&self, pos: Rank) -> &SearchResult {
        self.0[(pos as usize) - 1]
    }

    /// Number of clicked results from `Rank::First` to `pos`.
    fn cumulative_clicks(&self, pos: Rank) -> usize {
        self.0
            .iter()
            .filter(|r| r.relevance > ClickSat::Low && r.position <= pos)
            .count()
    }

    /// Rank of the result with the matching `url`.
    fn rank_of(&self, url: &str) -> Option<Rank> {
        self.0
            .iter()
            .find_map(|r| (r.url == url).then(|| r.position))
    }
}

struct DocAddr<'a> {
    url: UrlOrDom<'a>,
    dom: UrlOrDom<'a>,
}

impl<'a> DocAddr<'a> {
    fn new(url: &'a str, dom: &'a str) -> Self {
        Self {
            url: UrlOrDom::Url(url),
            dom: UrlOrDom::Dom(dom),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum UrlOrDom<'a> {
    /// A specific URL.
    Url(&'a str),
    /// Any URL belonging to the given domain.
    Dom(&'a str),
}

/// Query submission timescale.
#[derive(Clone, Copy)]
pub(crate) enum SessionCond {
    /// Before current session.
    Anterior(i32),
    /// Current session.
    Current(i32),
    /// All historic.
    All,
}

/// Filter predicate representing a boolean condition on a search result.
#[derive(Clone, Copy)]
pub(crate) struct FilterPred<'a> {
    doc: UrlOrDom<'a>,
    query: Option<i32>,
    session: SessionCond,
}

impl<'a> FilterPred<'a> {
    pub(crate) fn new(doc: UrlOrDom<'a>) -> Self {
        Self {
            doc,
            query: None,
            session: SessionCond::All,
        }
    }

    pub(crate) fn with_query(mut self, query_id: i32) -> Self {
        self.query = Some(query_id);
        self
    }

    pub(crate) fn with_session(mut self, session: SessionCond) -> Self {
        self.session = session;
        self
    }

    /// Lookup the atoms making up the aggregate feature for this filter predicate.
    ///
    /// Mapping of (predicate => atomic features) based on Dataiku's winning model (see paper).
    pub(crate) fn agg_atoms(&self) -> SmallVec<[AtomFeat; 6]> {
        use AtomFeat::{
            CondProb,
            MeanRecipRank as MRR,
            MeanRecipRankAll as mrr,
            SnippetQuality as SQ,
        };
        use MrrOutcome::{Click, Miss, Skip};
        use SessionCond::{All, Anterior as Ant, Current};
        use UrlOrDom::{Dom, Url};

        let skip = CondProb(ClickSat::Skip);
        let miss = CondProb(ClickSat::Miss);
        let click2 = CondProb(ClickSat::High);

        match (self.doc, self.query, self.session) {
            (Dom(_), None, All) => smallvec![skip, miss, click2, SQ],
            (Dom(_), None, Ant(_)) => smallvec![click2, miss, SQ],
            (Url(_), None, All) => smallvec![MRR(Click), click2, miss, SQ],
            (Url(_), None, Ant(_)) => smallvec![click2, miss, SQ],
            (Dom(_), Some(_), All) => smallvec![miss, SQ, MRR(Miss)],
            (Dom(_), Some(_), Ant(_)) => smallvec![SQ],
            (Url(_), Some(_), All) => smallvec![mrr, click2, miss, SQ],
            (Url(_), Some(_), Ant(_)) => smallvec![mrr, MRR(Click), MRR(Miss), MRR(Skip), skip, SQ],
            (Url(_), Some(_), Current(_)) => smallvec![MRR(Miss)],
            _ => smallvec![],
        }
    }

    /// Lookup the atoms making up the cumulated feature for this filter predicate.
    ///
    /// Mapping of (predicate => atomic features) based on Dataiku's winning model (see paper).
    /// Note that for cumulated features, the atoms are always conditional probabilities.
    pub(crate) fn cum_atoms(&self) -> SmallVec<[ClickSat; 3]> {
        use ClickSat::{High as click2, Medium as click1, Skip as skip};
        use SessionCond::All;
        use UrlOrDom::*;

        match (self.doc, self.query, self.session) {
            (Url(_), None, All) => smallvec![skip, click1, click2],
            _ => smallvec![],
        }
    }

    /// Applies the predicate to the given search result.
    pub(crate) fn apply(&self, r: impl AsRef<SearchResult>) -> bool {
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

/// Families of features for a non-personalised search result, based on Dataiku's specification.
struct Features {
    /// Unpersonalised rank.
    rank: Rank,
    /// Aggregate features.
    aggreg: AggregFeatures,
    /// User features.
    user: UserFeatures,
    /// Query features.
    query: QueryFeatures,
    /// Cumulated features.
    cum: CumFeatures,
    /// Terms variety count.
    terms_variety: usize,
    /// Weekend domain seasonality.
    seasonality: f32,
}

/// Build features for a user's search `res`ult given her past search `hist`ory.
fn build_features(hist: &[SearchResult], res: SearchResult) -> Features {
    let rank = res.position;
    let aggreg = aggreg_features(hist, &res);
    let user = user_features(hist);
    let query = query_features(hist, &res);
    let cum = cum_features(hist, &res);
    let terms_variety = terms_variety(hist, res.session_id);
    // NOTE according to Dataiku spec, this should be the weekend seasonality
    // factor when `res.day` is a weekend, otherwise the inverse (weekday
    // seasonality) factor. a bug in soundgarden sets this to always be weekend
    // seasonality but since the model is trained on it, we match that
    // behaviour here.
    let seasonality = seasonality(hist, res.domain);

    Features {
        rank,
        aggreg,
        user,
        query,
        cum,
        terms_variety,
        seasonality,
    }
}

/// Mean reciprocal rank of results filtered by outcome and a predicate.
///
/// It is defined as the ratio:
///```text
///   sum{1/r.position} + 0.283
/// ----------------------------
///    |rs(outcome, pred)| + 1
/// ```
/// where the sum ranges over each search result `r` in `rs`(`outcome`, `pred`),
/// i.e. satisfying `pred` and matching `outcome`.
///
/// The formula uses some form of additive smoothing with a prior 0.283 (see Dataiku paper).
pub(crate) fn mean_recip_rank(
    rs: &[impl AsRef<SearchResult>],
    outcome: Option<MrrOutcome>,
    pred: Option<FilterPred>,
) -> f32 {
    let filtered = rs
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
fn seasonality(history: &[SearchResult], domain: String) -> f32 {
    let (clicks_wknd, clicks_wkday) = history
        .iter()
        .filter(|r| r.domain == domain && r.relevance > ClickSat::Low)
        .fold((0, 0), |(wknd, wkday), r| {
            // NOTE weekend days should obviously be Sat/Sun but there is a bug
            // in soundgarden that effectively treats Thu/Fri as weekends
            // instead. since the model has been trained as such with the
            // soundgarden implementation, we match that behaviour here.
            if r.day == DayOfWeek::Thu || r.day == DayOfWeek::Fri {
                (wknd + 1, wkday)
            } else {
                (wknd, wkday + 1)
            }
        });

    2.5 * (1. + clicks_wknd as f32) / (1. + clicks_wkday as f32)
}

/// Entropy over the rank of the given results that were clicked.
pub(crate) fn click_entropy(results: &[impl AsRef<SearchResult>]) -> f32 {
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

/// Quality of the snippet associated with a search result.
///
/// Snippet quality is defined as:
/// ```text
///       sum{score(r)}
/// --------------------------
/// |hist({Miss, Skip}, pred)|
/// ```
/// where the sum ranges over all result sets containing a result `r` with URL matching that of `res`.
pub(crate) fn snippet_quality(hist: &[SearchResult], res: &SearchResult, pred: FilterPred) -> f32 {
    let pred_filtered = hist.iter().filter(|r| pred.apply(r));
    let denom = pred_filtered
        .clone()
        .filter(|r| r.relevance == ClickSat::Miss || r.relevance == ClickSat::Skip)
        .count() as f32;

    if denom == 0. {
        return 0.;
    }

    let numer = pred_filtered
        .group_by(|r| (r.session_id, r.query_counter))
        .into_iter()
        .filter_map(|(_, rs)| {
            let rs = ResultSet::new(rs.collect());
            rs.rank_of(&res.url).map(|pos| snippet_score(rs, pos))
        })
        .sum::<f32>();

    numer / denom
}

/// Scores the search result ranked at position `pos` in the result set `rs`.
///
/// The score(r) of a search result r is defined:
/// *  0           if r is a `Miss`
/// *  1 / p       if r is the pth clicked result of the page
/// * -1 / p_final if r is a `Skip`
///
/// As click ordering information is unavailable, assume it follows the ranking order.
fn snippet_score(rs: ResultSet, pos: Rank) -> f32 {
    match rs.get(pos).relevance {
        // NOTE unclear how to score Low, treat as a Miss
        ClickSat::Miss | ClickSat::Low => 0.,
        ClickSat::Skip => {
            let total_clicks = rs.cumulative_clicks(Rank::Last) as f32;
            if total_clicks == 0. {
                0.
            } else {
                -total_clicks.recip()
            }
        }
        _ => {
            let cum_clicks = rs.cumulative_clicks(pos) as f32;
            if cum_clicks == 0. {
                0.
            } else {
                cum_clicks.recip()
            }
        }
    }
}

/// Probability of an outcome conditioned on some predicate.
///
/// It is defined:
/// ```text
/// |hist(outcome, pred)| + prior(outcome)
/// --------------------------------------
///   |hist(pred)| + sum{prior(outcome')}
/// ```
/// The formula uses some form of additive smoothing with `prior(Miss)` = `1` and `0` otherwise.
/// See Dataiku paper. Note then the `sum` term amounts to `1`.
pub(crate) fn cond_prob(hist: &[SearchResult], outcome: ClickSat, pred: FilterPred) -> f32 {
    let hist_pred = hist.iter().filter(|r| pred.apply(r));
    let hist_pred_outcome = hist_pred.clone().filter(|r| r.relevance == outcome).count();

    // NOTE soundgarden implements conditional probabilities differently to
    // Dataiku's spec. nevertheless, since the model has been trained as such on
    // the soundgarden implementation, we match its behaviour here.
    let (numer, denom) = if let ClickSat::Miss = outcome {
        (hist_pred_outcome + 1, hist_pred.count() + hist_pred_outcome)
    } else {
        (hist_pred_outcome, hist_pred.count())
    };

    if denom == 0 {
        0.
    } else {
        numer as f32 / denom as f32
    }
}
