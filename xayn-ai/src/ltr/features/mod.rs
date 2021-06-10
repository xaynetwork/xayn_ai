//! Feature extraction algorithm based on Dataiku's solution to Yandex Personalized Web Search
//! Challenge [1]. See [2] for the first implementation in Python.
//!
//! [1]: https://www.academia.edu/9872579/Dataikus_Solution_to_Yandexs_Personalized_Web_Search_Challenge
//! [2]: https://github.com/xaynetwork/soundgarden

#![allow(dead_code)] // TEMP

mod aggregate;
mod cumulate;
mod query;
mod user;

use itertools::{izip, Itertools};
use ndarray::{Array2, Axis};
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;

use crate::{
    data::document_data::DocumentDataWithCoi,
    DayOfWeek,
    DocumentHistory,
    QueryId,
    Relevance,
    SessionId,
    UserAction,
};

use aggregate::AggregFeatures;
use cumulate::{CumFeatures, CumFeaturesAccumulator};
use query::QueryFeatures;
use user::UserFeatures;

/// Action from the user on a particular search result.
///
/// Based on Yandex notion of dwell-time: time elapsed between a click and the next action.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub(crate) enum Action {
    /// No click after examining snippet.
    Skip,
    /// No click without examining snippet.
    Miss,
    /// Less than 50 units of dwell-time or no click.
    Click0, // TODO consider removing later
    /// From 50 to 300 units of dwell-time.
    Click1,
    /// More than 300 units of dwell-time or last click of the session.
    Click2,
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
    Ninth,
    Last,
}

impl From<Rank> for f32 {
    fn from(rank: Rank) -> Self {
        rank as u8 as f32
    }
}

#[derive(PartialEq, Eq, Hash)]
/// Search query.
pub(crate) struct Query {
    /// Session identifier.
    pub(crate) session_id: SessionId,
    /// Query count within session.
    pub(crate) query_count: usize,
    /// Query identifier.
    pub(crate) query_id: QueryId,
    /// Words of the query.
    pub(crate) query_words: Vec<String>,
}

/// A result from a new search performed by the user.
pub struct DocSearchResult {
    /// Search query.
    pub(crate) query: Query,
    /// URL of result.
    pub(crate) url: String,
    /// Domain of result.
    pub(crate) domain: String,
    /// Initial unpersonalised rank among other results of the search.
    pub(crate) init_rank: Rank,
}

impl AsRef<DocSearchResult> for DocSearchResult {
    fn as_ref(&self) -> &DocSearchResult {
        self
    }
}

/// A reranked result from a search performed at some point in the user's history.
pub struct HistSearchResult {
    /// Search query.
    pub(crate) query: Query,
    /// URL of result.
    pub(crate) url: String,
    /// Domain of result.
    pub(crate) domain: String,
    /// Reranked position among other results of the search.
    pub(crate) rerank: Rank,
    /// Day of week search was performed.
    pub(crate) day: DayOfWeek,
    /// User action on this result.
    pub(crate) action: Action,
}

impl AsRef<HistSearchResult> for HistSearchResult {
    fn as_ref(&self) -> &HistSearchResult {
        self
    }
}

impl HistSearchResult {
    pub(crate) fn is_clicked(&self) -> bool {
        self.action >= Action::Click1
    }

    pub(crate) fn is_skipped(&self) -> bool {
        self.action == Action::Skip
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
    CondProb(Action),
    /// Snippet quality.
    SnippetQuality,
}

pub(crate) type FeatMap = HashMap<AtomFeat, f32>;

#[derive(Clone)]
/// Search results from some query.
struct ResultSet<'a>(Vec<&'a HistSearchResult>);

impl<'a> ResultSet<'a> {
    /// New result set.
    ///
    /// Assumes `rs[i]` contains the search result `r` with `r.position` `i+1`.
    fn new(rs: Vec<&'a HistSearchResult>) -> Self {
        Self(rs)
    }

    /// Iterates over all documents by in ascending ranking order.
    fn documents(&self) -> impl Iterator<Item = &'a HistSearchResult> + '_ {
        self.0.iter().copied()
    }

    /// Iterates over all documents which have been clicked in ascending ranking order.
    fn clicked_documents(&self) -> impl Iterator<Item = &'a HistSearchResult> + '_ {
        self.0.iter().flat_map(|doc| doc.is_clicked().then(|| *doc))
    }

    /// Search result at position `pos` of the result set.
    ///
    /// # Panic
    /// Panics if the length of the underlying vector is less than `pos`.
    fn get(&self, pos: Rank) -> &HistSearchResult {
        self.0[(pos as usize) - 1]
    }

    /// Number of clicked results from `Rank::First` to `pos`.
    fn cumulative_clicks(&self, pos: Rank) -> usize {
        self.0
            .iter()
            .filter(|r| r.action > Action::Click0 && r.rerank <= pos)
            .count()
    }

    /// Rank of the result with the matching `url`.
    fn rank_of(&self, url: &str) -> Option<Rank> {
        self.0.iter().find_map(|r| (r.url == url).then(|| r.rerank))
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
    Anterior(SessionId),
    /// Current session.
    Current(SessionId),
    /// All historic.
    All,
}

/// Filter predicate representing a boolean condition on a search result.
#[derive(Clone, Copy)]
pub(crate) struct FilterPred<'a> {
    doc: UrlOrDom<'a>,
    query: Option<QueryId>,
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

    pub(crate) fn with_query(mut self, query_id: QueryId) -> Self {
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

        let skip = CondProb(Action::Skip);
        let miss = CondProb(Action::Miss);
        let click2 = CondProb(Action::Click2);

        match (self.doc, self.query, self.session) {
            (Dom(_), None, All) => smallvec![skip, miss, click2, SQ],
            (Dom(_), None, Ant(_)) => smallvec![click2, miss, SQ],
            (Url(_), None, All) => smallvec![MRR(Click), click2, miss, SQ],
            (Url(_), None, Ant(_)) => smallvec![click2, miss, SQ],
            (Dom(_), Some(_), All) => smallvec![miss, SQ, MRR(Miss)],
            (Dom(_), Some(_), Ant(_)) => smallvec![SQ],
            (Url(_), Some(_), All) => smallvec![mrr, click2, miss, SQ],
            (Url(_), Some(_), Ant(_)) => smallvec![mrr, MRR(Click), MRR(Miss), MRR(Skip), miss, SQ],
            (Url(_), Some(_), Current(_)) => smallvec![MRR(Miss)],
            _ => smallvec![],
        }
    }

    /// Lookup the atoms making up the cumulated feature for this filter predicate.
    ///
    /// Mapping of (predicate => atomic features) based on Dataiku's winning model (see paper).
    /// Note that for cumulated features, the atoms are always conditional probabilities.
    pub(crate) fn cum_atoms(&self) -> SmallVec<[Action; 3]> {
        use Action::{Click1 as click1, Click2 as click2, Skip as skip};
        use SessionCond::All;
        use UrlOrDom::*;

        match (self.doc, self.query, self.session) {
            (Url(_), None, All) => smallvec![skip, click1, click2],
            _ => smallvec![],
        }
    }

    /// Applies the predicate to the given search result.
    pub(crate) fn apply(&self, r: impl AsRef<HistSearchResult>) -> bool {
        let r = r.as_ref();
        let doc_cond = match self.doc {
            UrlOrDom::Url(url) => r.url == url,
            UrlOrDom::Dom(dom) => r.domain == dom,
        };
        let query_cond = match self.query {
            Some(id) => r.query.query_id == id,
            None => true,
        };
        let session_id = r.query.session_id;
        let session_cond = match self.session {
            // `id` is of the *current* session hence any other must be *older*
            SessionCond::Anterior(id) => session_id != id,
            SessionCond::Current(id) => session_id == id,
            SessionCond::All => true,
        };
        doc_cond && query_cond && session_cond
    }
}

// TODO mention diffs
/// Families of features for a non-personalised search result, based on Dataiku's specification.
pub(crate) struct Features {
    /// Unpersonalised rank.
    init_rank: Rank,
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

impl Features {
    /// Builds features for a new search result `doc` from the user given her past search history `hists`.
    ///
    /// `cum_acc` contains the cumulative features of other results in the search ranked above `doc`.
    fn build(
        hists: &[HistSearchResult],
        doc: &DocSearchResult,
        cum_acc: &mut CumFeaturesAccumulator,
    ) -> Self {
        let init_rank = doc.init_rank;
        let aggreg = AggregFeatures::build(hists, doc);
        let user = UserFeatures::build(hists);
        let query = QueryFeatures::build(hists, doc);
        let cum = cum_acc.build_next(hists, doc);
        let terms_variety = terms_variety(doc);
        let seasonality = seasonality(hists, doc);

        Self {
            init_rank,
            aggreg,
            user,
            query,
            cum,
            terms_variety,
            seasonality,
        }
    }
}

/// Builds features for each new search result in `docs`.
///
/// `docs` must be in order of rank, starting from `Rank::First`.
pub(crate) fn build_features(
    hists: &[HistSearchResult],
    docs: &[DocSearchResult],
) -> Vec<Features> {
    let mut cum_acc = CumFeaturesAccumulator::new();
    docs.iter()
        .map(|doc| Features::build(hists, doc, &mut cum_acc))
        .collect()
}

/// Converts a sequence of `Feature`s into a 2-dimensional ndarray.
pub(crate) fn features_to_ndarray(feats_list: &[Features]) -> Array2<f32> {
    let mut arr = Array2::zeros((feats_list.len(), 50));

    let click_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Click);
    let miss_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Miss);
    let skip_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Skip);
    let combine_mrr = &AtomFeat::MeanRecipRankAll;
    let click2 = &AtomFeat::CondProb(Action::Click2);
    let missed = &AtomFeat::CondProb(Action::Miss);
    let skipped = &AtomFeat::CondProb(Action::Skip);
    let snippet_quality = &AtomFeat::SnippetQuality;

    for (feats, mut row) in izip!(feats_list, arr.axis_iter_mut(Axis(0))) {
        let Features {
            init_rank,
            aggreg,
            user,
            query,
            cum,
            terms_variety,
            seasonality,
        } = feats;

        row[0] = init_rank.to_owned().into();
        row[1] = aggreg.url[click_mrr];
        row[2] = aggreg.url[click2];
        row[3] = aggreg.url[missed];
        row[4] = aggreg.url[snippet_quality];
        row[5] = aggreg.url_ant[click2];
        row[6] = aggreg.url_ant[missed];
        row[7] = aggreg.url_ant[snippet_quality];
        row[8] = aggreg.url_query[combine_mrr];
        row[9] = aggreg.url_query[click2];
        row[10] = aggreg.url_query[missed];
        row[11] = aggreg.url_query[snippet_quality];
        row[12] = aggreg.url_query_ant[combine_mrr];
        row[13] = aggreg.url_query_ant[click_mrr];
        row[14] = aggreg.url_query_ant[miss_mrr];
        row[15] = aggreg.url_query_ant[skip_mrr];
        row[16] = aggreg.url_query_ant[missed];
        row[17] = aggreg.url_query_ant[snippet_quality];
        row[18] = aggreg.url_query_curr[miss_mrr];
        row[19] = aggreg.dom[skipped];
        row[20] = aggreg.dom[missed];
        row[21] = aggreg.dom[click2];
        row[22] = aggreg.dom[snippet_quality];
        row[23] = aggreg.dom_ant[click2];
        row[24] = aggreg.dom_ant[missed];
        row[25] = aggreg.dom_ant[snippet_quality];
        row[26] = aggreg.dom_query[missed];
        row[27] = aggreg.dom_query[snippet_quality];
        row[28] = aggreg.dom_query[miss_mrr];
        row[29] = aggreg.dom_query_ant[snippet_quality];
        row[30] = query.click_entropy;
        row[31] = query.num_terms as f32;
        row[32] = query.mean_query_count;
        row[33] = query.occurs_per_session;
        row[34] = query.num_occurs as f32;
        row[35] = query.click_mrr;
        row[36] = query.mean_clicks;
        row[37] = query.mean_skips;
        row[38] = user.click_entropy;
        row[39] = user.click_counts.click12 as f32;
        row[40] = user.click_counts.click345 as f32;
        row[41] = user.click_counts.click6up as f32;
        row[42] = user.num_queries as f32;
        row[43] = user.words_per_query;
        row[44] = user.words_per_session;
        row[45] = cum.url_skip;
        row[46] = cum.url_click1;
        row[47] = cum.url_click2;
        row[48] = *terms_variety as f32;
        row[49] = *seasonality;
    }
    arr
}

impl From<Features> for [f32; 50] {
    fn from(feats: Features) -> Self {
        let click_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Click);
        let miss_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Miss);
        let skip_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Skip);
        let combine_mrr = &AtomFeat::MeanRecipRankAll;
        let click2 = &AtomFeat::CondProb(Action::Click2);
        let missed = &AtomFeat::CondProb(Action::Miss);
        let skipped = &AtomFeat::CondProb(Action::Skip);
        let snippet_quality = &AtomFeat::SnippetQuality;

        let Features {
            init_rank,
            aggreg,
            user,
            query,
            cum,
            terms_variety,
            seasonality,
        } = feats;

        [
            init_rank.into(),
            aggreg.url[click_mrr],
            aggreg.url[click2],
            aggreg.url[missed],
            aggreg.url[snippet_quality],
            aggreg.url_ant[click2],
            aggreg.url_ant[missed],
            aggreg.url_ant[snippet_quality],
            aggreg.url_query[combine_mrr],
            aggreg.url_query[click2],
            aggreg.url_query[missed],
            aggreg.url_query[snippet_quality],
            aggreg.url_query_ant[combine_mrr],
            aggreg.url_query_ant[click_mrr],
            aggreg.url_query_ant[miss_mrr],
            aggreg.url_query_ant[skip_mrr],
            aggreg.url_query_ant[missed],
            aggreg.url_query_ant[snippet_quality],
            aggreg.url_query_curr[miss_mrr],
            aggreg.dom[skipped],
            aggreg.dom[missed],
            aggreg.dom[click2],
            aggreg.dom[snippet_quality],
            aggreg.dom_ant[click2],
            aggreg.dom_ant[missed],
            aggreg.dom_ant[snippet_quality],
            aggreg.dom_query[missed],
            aggreg.dom_query[snippet_quality],
            aggreg.dom_query[miss_mrr],
            aggreg.dom_query_ant[snippet_quality],
            query.click_entropy,
            query.num_terms as f32,
            query.mean_query_count,
            query.occurs_per_session,
            query.num_occurs as f32,
            query.click_mrr,
            query.mean_clicks,
            query.mean_skips,
            user.click_entropy,
            user.click_counts.click12 as f32,
            user.click_counts.click345 as f32,
            user.click_counts.click6up as f32,
            user.num_queries as f32,
            user.words_per_query,
            user.words_per_session,
            cum.url_skip,
            cum.url_click1,
            cum.url_click2,
            terms_variety as f32,
            seasonality,
        ]
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
    rs: &[impl AsRef<HistSearchResult>],
    outcome: Option<MrrOutcome>,
    pred: Option<FilterPred>,
) -> f32 {
    let filtered = rs
        .iter()
        .filter(|r| {
            let relevance = r.as_ref().action;
            match outcome {
                Some(MrrOutcome::Miss) => relevance == Action::Miss,
                Some(MrrOutcome::Skip) => relevance == Action::Skip,
                Some(MrrOutcome::Click) => relevance > Action::Click0,
                None => true,
            }
        })
        .filter(|r| pred.map_or(true, |p| p.apply(r)))
        .collect_vec();

    let denom = 1. + filtered.len() as f32;
    let numer = 0.283 // prior recip rank assuming uniform distributed ranks
        + filtered
            .into_iter()
            .map(|r| f32::from(r.as_ref().rerank).recip())
            .sum::<f32>();

    numer / denom
}

/// Counts the variety of terms over a given test session.
///
/// # Implementation Differences
///
/// Soundgarden uses the "query_array" instead of the current session
///
/// While this is clearly a bug, we need to be consistent with it. In turn we
/// don't need to filter for the session id as all current results are from the
/// current search query and as such the current session.
///
/// Additionally this also causes all inspected results/documents to have the
/// same query words and in turn this is just the number of unique words in the
/// current query, which is kinda very pointless.
fn terms_variety(doc: &DocSearchResult) -> usize {
    doc.query.query_words.iter().unique().count()
}

/// Weekend seasonality factor of a given domain.
///
/// # Implementation Differences
///
/// According to Dataiku spec, this should be the *weekend* seasonality when `doc.day` is a weekend
/// otherwise the inverse (*weekday* seasonality). A bug in soundgarden sets this to always be
/// weekend seasonality but since the model is trained on it, we match that behaviour here.
///
/// If there are no matching entries for the given domain then `0` is returned instead of the
/// expected 2.5. Again, we do this here to be in sync with soundgarden.
fn seasonality(history: &[HistSearchResult], doc: &DocSearchResult) -> f32 {
    let (clicks_wknd, clicks_wkday) = history
        .iter()
        .filter(|hist| hist.domain == doc.domain && hist.is_clicked())
        .fold((0, 0), |(wknd, wkday), hist| {
            // NOTE weekend days should obviously be Sat/Sun but there is a bug
            // in soundgarden that effectively treats Thu/Fri as weekends
            // instead. since the model has been trained as such with the
            // soundgarden implementation, we match that behaviour here.
            if hist.day == DayOfWeek::Thu || hist.day == DayOfWeek::Fri {
                (wknd + 1, wkday)
            } else {
                (wknd, wkday + 1)
            }
        });

    if clicks_wknd + clicks_wkday == 0 {
        0.0
    } else {
        2.5 * (1. + clicks_wknd as f32) / (1. + clicks_wkday as f32)
    }
}

/// Entropy over the rank of the given results that were clicked.
pub(crate) fn click_entropy(results: &[impl AsRef<HistSearchResult>]) -> f32 {
    let rank_freqs = results
        .iter()
        .filter_map(|r| {
            let r = r.as_ref();
            (r.action > Action::Click0).then(|| r.rerank)
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
///       sum{score(query)}
/// --------------------------
/// |hist({Miss, Skip}, pred)|
/// ```
///
/// where the sum ranges over all query results containing a result `r` with URL/domain matching the URL/domain of `res`.
///
/// If `|hist({Miss, Skip}, pred)|` is `0` then `0` is returned.
pub(crate) fn snippet_quality(hists: &[HistSearchResult], pred: FilterPred) -> f32 {
    let miss_skip_count = hists
        .iter()
        .filter(|hist| {
            pred.apply(hist) && (hist.action == Action::Miss || hist.action == Action::Skip)
        })
        .count() as f32;

    if miss_skip_count == 0. {
        return 0.;
    }

    let total_score = hists
        .iter()
        .group_by(|hist| (hist.query.session_id, hist.query.query_count))
        .into_iter()
        .filter_map(|(_, hs)| {
            let rs = ResultSet::new(hs.into_iter().collect());
            let has_match = rs.documents().any(|doc| pred.apply(doc));
            has_match.then(|| snippet_score(&rs, pred))
        })
        .sum::<f32>();

    total_score / miss_skip_count
}

/// Scores the search query result.
///
/// The base score of `0` is modified in up to two ways before being returned:
///
/// 1.  `1 / p` is added if there exist a clicked document matching `pred`. `p` is the number of
///     clicked documents which do not match `pred` and have a better ranking then the found
///     matching document.
///
/// 2. `-1 / nr_clicked` is added if there exists a skipped document matching `pred`. `nr_clicked`
///     is the number of all clicked documents independent of weather or not they match.
fn snippet_score(res: &ResultSet, pred: FilterPred) -> f32 {
    let mut score = 0.0;

    if res
        .documents()
        .any(|doc| pred.apply(doc) && doc.is_skipped())
    {
        let total_clicks = res.clicked_documents().count() as f32;
        if total_clicks != 0. {
            score -= total_clicks.recip();
        }
    }

    if let Some(cum_clicks_before_match) = res.clicked_documents().position(|doc| pred.apply(doc)) {
        score += (cum_clicks_before_match as f32 + 1.).recip();
    }

    score
}

/// Probability of an outcome conditioned on some predicate.
///
/// It is defined:
///
/// ```text
/// |hist(outcome, pred)| + prior(outcome)
/// --------------------------------------
///   |hist(pred)| + sum{prior(outcome')}
/// ```
///
/// The formula uses some form of additive smoothing with `prior(Miss)` = `1` and `0` otherwise.
/// See Dataiku paper. Note then the `sum` term amounts to `1`.
///
/// # Implementation Mismatch
///
/// The python implementation (soundgarden) on which the model was trained does not match dataiku's
/// definition above. Instead:
///
/// For every `outcome` except `ClickSat::Miss` the following formula is used:
///
/// ```text
/// |hist(outcome, pred)|
/// ----------------------
///    |hist(pred)|
/// ```
///
/// If the `outcome` is `ClickSet::Miss` then the following formula is used:
///
/// ```text
///       |hist(outcome, pred)| + 1
/// ---------------------------------------
/// |hist(outcome, pred)| + |hist(pred)|
/// ```
///
/// In both cases it defaults to 0 if the denominator is 0 (as in this case the
/// numerator should be 0 too).
pub(crate) fn cond_prob(hist: &[HistSearchResult], outcome: Action, pred: FilterPred) -> f32 {
    let hist_pred = hist.iter().filter(|r| pred.apply(r));
    let hist_pred_outcome = hist_pred.clone().filter(|r| r.action == outcome).count();

    let (numer, denom) = if let Action::Miss = outcome {
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
