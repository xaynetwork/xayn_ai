//! Feature extraction algorithm based on Dataiku's solution to Yandex Personalized Web Search
//! Challenge [1]. See [2] for the first implementation in Python.
//!
//! [1]: https://www.academia.edu/9872579/Dataikus_Solution_to_Yandexs_Personalized_Web_Search_Challenge
//! [2]: https://github.com/xaynetwork/soundgarden

mod aggregate;
mod cumulate;
mod query;
mod user;

use derive_more::From;
use itertools::{izip, Itertools};
use ndarray::{array, Array2, Axis};
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;
use thiserror::Error;

use crate::{
    data::document_data::DocumentDataWithCoi,
    DayOfWeek,
    DocumentHistory,
    QueryId,
    Relevance,
    SessionId,
    UserAction,
};

use aggregate::AggregateFeatures;
use cumulate::{CumFeaturesAccumulator, CumulatedFeatures};
use query::QueryFeatures;
use user::UserFeatures;

/// Action from the user on a particular search result.
///
/// Based on Yandex notion of dwell-time: time elapsed between a click and the next action.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
#[cfg_attr(test, derive(Debug))]
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

/// `Rank` n represents a ranking of n + 1.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, From)]
pub(crate) struct Rank(usize);

impl From<Rank> for f32 {
    fn from(rank: Rank) -> Self {
        (rank.0 + 1) as f32
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(Clone))]
/// Search query.
pub(crate) struct Query {
    /// Session identifier.
    pub(crate) session_id: SessionId,
    /// Query count within session.
    pub(crate) query_count: usize,
    /// Identifier unique to a specific sequence of query words.
    ///
    /// Any query with the same query words MUST HAVE the
    /// same query id for the feature extraction to work
    /// correctly.
    pub(crate) query_id: QueryId,
    /// Words of the query.
    pub(crate) query_words: Vec<String>,
}

#[derive(Debug)]
/// A result from a new search performed by the user.
pub struct DocSearchResult {
    /// Search query.
    pub(crate) query: Query,
    /// URL of result.
    pub(crate) url: String,
    /// Domain of result.
    pub(crate) domain: String,
    /// Initial unpersonalised rank among other results of the search.
    pub(crate) initial_rank: Rank,
}

impl From<&DocumentDataWithCoi> for DocSearchResult {
    fn from(doc_data: &DocumentDataWithCoi) -> Self {
        let initial_rank = doc_data.document_base.initial_ranking.into();
        let content = &doc_data.document_content;
        let query_words = content.query_words.split_whitespace().map_into().collect();

        DocSearchResult {
            query: Query {
                session_id: content.session,
                query_count: content.query_count,
                query_id: content.query_id,
                query_words,
            },
            url: content.url.clone(),
            domain: content.domain.clone(),
            initial_rank,
        }
    }
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
    /// Final reranked position among other results of the search.
    pub(crate) final_rank: Rank,
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

impl From<&DocumentHistory> for HistSearchResult {
    fn from(doc_hist: &DocumentHistory) -> Self {
        let final_rank = doc_hist.rank.into();
        let query_words = doc_hist.query_words.split_whitespace().map_into().collect();
        let action = match (doc_hist.user_action, doc_hist.relevance) {
            (UserAction::Miss, _) => Action::Miss,
            (UserAction::Skip, _) => Action::Skip,
            (UserAction::Click, Relevance::Low) => Action::Click0,
            (UserAction::Click, Relevance::Medium) => Action::Click1,
            (UserAction::Click, Relevance::High) => Action::Click2,
        };

        HistSearchResult {
            query: Query {
                session_id: doc_hist.session,
                query_count: doc_hist.query_count,
                query_id: doc_hist.query_id,
                query_words,
            },
            url: doc_hist.url.clone(),
            domain: doc_hist.domain.clone(),
            final_rank,
            day: doc_hist.day,
            action,
        }
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
#[cfg_attr(test, derive(Debug))]
pub(crate) enum MrrOutcome {
    Miss,
    Skip,
    Click,
}

/// Atomic features of which an aggregate feature is composed of.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(Debug))]
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
    /// Assumes `rs` is ordered by rank starting from Rank(0).
    fn new(rs: Vec<&'a HistSearchResult>) -> Self {
        Self(rs)
    }

    /// Iterates over all documents by in ascending ranking order.
    fn documents(&self) -> impl Iterator<Item = &'a HistSearchResult> + '_ {
        self.0.iter().copied()
    }

    /// Iterates over all documents which have been clicked in ascending ranking order.
    fn clicked_documents(&self) -> impl Iterator<Item = &'a HistSearchResult> + '_ {
        self.0
            .iter()
            .flat_map(|hist| hist.is_clicked().then(|| *hist))
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug))]
pub(crate) enum UrlOrDom<'a> {
    /// A specific URL.
    Url(&'a str),
    /// Any URL belonging to the given domain.
    Dom(&'a str),
}

/// Query submission timescale.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug))]
pub(crate) enum SessionCond {
    /// Not the current Session
    ///
    /// # Implementation difference to Soundgarden.
    ///
    /// Soundgarden filters by "all previous session",
    /// but in our case the current session is always the
    /// last session, furthermore we do not have a ordering
    /// between sessions as they now use UUIDv4 ids instead
    /// of incremental counter ids.
    NotCurrent { current: SessionId },

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
        use SessionCond::{All, Current, NotCurrent};
        use UrlOrDom::{Dom, Url};

        let skip = CondProb(Action::Skip);
        let miss = CondProb(Action::Miss);
        let click2 = CondProb(Action::Click2);

        match (self.doc, self.query, self.session) {
            (Dom(_), None, All) => smallvec![skip, miss, click2, SQ],
            (Dom(_), None, NotCurrent { .. }) => smallvec![click2, miss, SQ],
            (Url(_), None, All) => smallvec![MRR(Click), click2, miss, SQ],
            (Url(_), None, NotCurrent { .. }) => smallvec![click2, miss, SQ],
            (Dom(_), Some(_), All) => smallvec![miss, SQ, MRR(Miss)],
            (Dom(_), Some(_), NotCurrent { .. }) => smallvec![SQ],
            (Url(_), Some(_), All) => smallvec![mrr, click2, miss, SQ],
            (Url(_), Some(_), NotCurrent { .. }) => {
                smallvec![mrr, MRR(Click), MRR(Miss), MRR(Skip), miss, SQ]
            }
            (Url(_), Some(_), Current(_)) => smallvec![MRR(Miss)],
            _ => smallvec![],
        }
    }

    /// Applies the predicate to the given search result.
    pub(crate) fn apply(&self, hist: impl AsRef<HistSearchResult>) -> bool {
        let hist = hist.as_ref();
        let doc_cond = match self.doc {
            UrlOrDom::Url(url) => hist.url == url,
            UrlOrDom::Dom(dom) => hist.domain == dom,
        };
        let query_cond = match self.query {
            Some(id) => hist.query.query_id == id,
            None => true,
        };
        let session_id = hist.query.session_id;
        let session_cond = match self.session {
            // `id` is of the *current* session hence any other must be *older*
            SessionCond::NotCurrent { current } => session_id != current,
            SessionCond::Current(id) => session_id == id,
            SessionCond::All => true,
        };
        doc_cond && query_cond && session_cond
    }
}

#[derive(Debug, Error)]
/// Features building error.
pub(crate) enum FeaturesError {
    #[error("Duplicate ranks in search results: {0:?}")]
    /// Search results contain duplicate ranks.
    DuplicateRanks(Vec<DocSearchResult>),
    #[error("Missing ranks in search results: {0:?}")]
    /// Search results contain missing ranks.
    MissingRanks(Vec<DocSearchResult>),
    #[error("Not all search results share the same query: {0:?}")]
    /// Search results contain a query mismatch.
    QueryMismatch(Vec<DocSearchResult>),
}

/// Families of features for a non-personalised search result, based on Dataiku's specification.
///
/// Note for our purposes, we adopt only a subset of Dataiku's features. In particular we omit:
/// * No. times the user performed the query (redundant)
/// * All 3 of the "any user" aggregate features (not supported by search app)
/// * 2 cumulated features: same domain & query and same url & query (not supported by search app)
/// * Collaborative filtering SVD (marginal benefit)
pub(crate) struct Features {
    /// Initial unpersonalised rank.
    initial_rank: Rank,
    /// Aggregate features.
    aggregated: AggregateFeatures,
    /// User features.
    user: UserFeatures,
    /// Query features.
    query: QueryFeatures,
    /// Cumulated features.
    cumulated: CumulatedFeatures,
    /// Terms variety count.
    terms_variety: usize,
    /// Weekend domain seasonality.
    seasonality: f32,
}

impl Features {
    /// Builds features for a new search result `doc` from the user given her past search history `hists`.
    ///
    /// `cum_acc` contains the cumulative features of other results ranked above `doc` in the search.
    fn build(
        hists: &[HistSearchResult],
        doc: &DocSearchResult,
        user: UserFeatures,
        query: QueryFeatures,
        cum_acc: &mut CumFeaturesAccumulator,
        terms_variety: usize,
    ) -> Self {
        let initial_rank = doc.initial_rank;
        let aggregate = AggregateFeatures::build(hists, doc);
        let cumulated = cum_acc.build_next(hists, doc);
        let seasonality = seasonality(hists, &doc.domain);

        Self {
            initial_rank,
            aggregated: aggregate,
            user,
            query,
            cumulated,
            terms_variety,
            seasonality,
        }
    }
}

/// Builds features for each new search result in `docs`.
///
/// If `docs` is not already in consecutive order of rank, i.e. 0, 1, 2, ...,
/// an attempt is made to sort `docs` so that it is.
///
/// # Errors
/// Fails if `docs` contains duplicate ranks, missing ranks, or a mismatched query.
pub(crate) fn build_features(
    hists: Vec<HistSearchResult>,
    docs: Vec<DocSearchResult>,
) -> Result<Vec<Features>, FeaturesError> {
    let len = docs.len();
    if len == 0 {
        return Ok(vec![]);
    };

    let docs_sorted = docs
        .iter()
        .sorted_by_key(|doc| doc.initial_rank)
        .dedup_by(|doc1, doc2| doc1.initial_rank == doc2.initial_rank)
        .collect_vec();

    if docs_sorted.len() != len {
        return Err(FeaturesError::DuplicateRanks(docs));
    };
    if docs_sorted[len - 1].initial_rank.0 != len - 1 {
        return Err(FeaturesError::MissingRanks(docs));
    };

    let q = &docs_sorted[0].query;
    let same_query = docs_sorted.iter().all(|doc| doc.query == *q);
    if !same_query {
        return Err(FeaturesError::QueryMismatch(docs));
    }

    let query = QueryFeatures::build(&hists, q);
    let user = UserFeatures::build(&hists);
    let mut cum_acc = CumFeaturesAccumulator::new();
    let tv = terms_variety(q);

    Ok(docs_sorted
        .into_iter()
        .map(|doc| Features::build(&hists, doc, user.clone(), query.clone(), &mut cum_acc, tv))
        .collect())
}

/// Converts a sequence of `Feature`s into a 2-dimensional ndarray.
///
/// Note that we follow the ordering of features as implemented in soundgarden,
/// since the ListNet model has been trained on that.
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

    for (feats, row) in izip!(feats_list, arr.axis_iter_mut(Axis(0))) {
        let Features {
            initial_rank,
            aggregated: aggregate,
            user,
            query,
            cumulated,
            terms_variety,
            seasonality,
        } = feats;

        array![
            initial_rank.to_owned().into(),
            aggregate.url[click_mrr],
            aggregate.url[click2],
            aggregate.url[missed],
            aggregate.url[snippet_quality],
            aggregate.url_ant[click2],
            aggregate.url_ant[missed],
            aggregate.url_ant[snippet_quality],
            aggregate.url_query[combine_mrr],
            aggregate.url_query[click2],
            aggregate.url_query[missed],
            aggregate.url_query[snippet_quality],
            aggregate.url_query_ant[combine_mrr],
            aggregate.url_query_ant[click_mrr],
            aggregate.url_query_ant[miss_mrr],
            aggregate.url_query_ant[skip_mrr],
            aggregate.url_query_ant[missed],
            aggregate.url_query_ant[snippet_quality],
            aggregate.url_query_curr[miss_mrr],
            aggregate.dom[skipped],
            aggregate.dom[missed],
            aggregate.dom[click2],
            aggregate.dom[snippet_quality],
            aggregate.dom_ant[click2],
            aggregate.dom_ant[missed],
            aggregate.dom_ant[snippet_quality],
            aggregate.dom_query[missed],
            aggregate.dom_query[snippet_quality],
            aggregate.dom_query[miss_mrr],
            aggregate.dom_query_ant[snippet_quality],
            query.click_entropy,
            query.num_terms as f32,
            query.mean_query_count,
            query.mean_occurs_per_session,
            query.num_occurs as f32,
            query.click_mrr,
            query.mean_clicks,
            query.mean_non_clicks,
            user.click_entropy,
            user.click_counts.click12 as f32,
            user.click_counts.click345 as f32,
            user.click_counts.click6up as f32,
            user.num_queries as f32,
            user.mean_words_per_query,
            user.mean_unique_words_per_session,
            cumulated.url_skip,
            cumulated.url_click1,
            cumulated.url_click2,
            *terms_variety as f32,
            *seasonality,
        ]
        .move_into(row);
    }
    arr
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
    hists: &[impl AsRef<HistSearchResult>],
    outcome: Option<MrrOutcome>,
    pred: Option<FilterPred>,
) -> f32 {
    let filtered = hists
        .iter()
        .filter(|hist| {
            let action = hist.as_ref().action;
            match outcome {
                Some(MrrOutcome::Miss) => action == Action::Miss,
                Some(MrrOutcome::Skip) => action == Action::Skip,
                Some(MrrOutcome::Click) => action >= Action::Click1,
                None => true,
            }
        })
        .filter(|hist| pred.map_or(true, |p| p.apply(hist)))
        .collect_vec();

    let denom = 1. + filtered.len() as f32;
    let numer = 0.283 // prior recip rank assuming uniform distributed ranks
        + filtered
            .into_iter()
            .map(|hist| f32::from(hist.as_ref().final_rank).recip())
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
fn terms_variety(query: &Query) -> usize {
    query.query_words.iter().unique().count()
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
fn seasonality(hists: &[HistSearchResult], domain: &str) -> f32 {
    let (clicks_wknd, clicks_wkday) = hists
        .iter()
        .filter(|hist| hist.domain == domain && hist.is_clicked())
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
pub(crate) fn click_entropy(hists: &[impl AsRef<HistSearchResult>]) -> f32 {
    let rank_freqs = hists
        .iter()
        .filter_map(|hist| {
            let hist = hist.as_ref();
            hist.is_clicked().then(|| hist.final_rank)
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
fn snippet_score(rs: &ResultSet, pred: FilterPred) -> f32 {
    let mut score = 0.0;

    if rs
        .documents()
        .any(|hist| pred.apply(hist) && hist.is_skipped())
    {
        let total_clicks = rs.clicked_documents().count() as f32;
        if total_clicks != 0. {
            score -= total_clicks.recip();
        }
    }

    if let Some(cum_clicks_before_match) = rs.clicked_documents().position(|doc| pred.apply(doc)) {
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
/// For every `outcome` except `Action::Miss` the following formula is used:
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
pub(crate) fn cond_prob(hists: &[HistSearchResult], outcome: Action, pred: FilterPred) -> f32 {
    let hist_pred = hists.iter().filter(|hist| pred.apply(hist));
    let hist_pred_outcome = hist_pred
        .clone()
        .filter(|hist| hist.action == outcome)
        .count();

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

#[cfg(test)]
mod tests {
    use std::{convert::TryFrom, path::Path};

    use once_cell::sync::Lazy;
    use serde::Deserialize;

    use crate::utils::mock_uuid;

    use super::*;

    fn history_by_url<'a>(
        iter: impl IntoIterator<Item = &'a (Action, &'a str)>,
    ) -> Vec<HistSearchResult> {
        let mut queries = HashMap::new();

        iter.into_iter()
            .enumerate()
            .map(|(id, (action, url))| {
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                HistSearchResult {
                    query: queries
                        .entry(per_query_id)
                        .or_insert_with(|| Query {
                            session_id: SessionId(mock_uuid(1)),
                            query_count: per_query_id,
                            query_id: QueryId(mock_uuid(per_query_id)),
                            query_words: vec![
                                "1".to_owned(),
                                "2".to_owned(),
                                per_query_id.to_string(),
                            ],
                        })
                        .clone(),
                    url: (*url).to_owned(),
                    domain: "example.com".to_owned(),
                    final_rank: Rank(in_query_id),
                    day: DayOfWeek::Tue,
                    action: *action,
                }
            })
            .collect()
    }

    #[test]
    fn test_cond_prob() {
        let history = history_by_url(&[
            (Action::Click1, "a/b"),
            (Action::Click2, "a/b"),
            (Action::Click1, "d.f"),
            (Action::Click1, "r"),
            (Action::Click1, "a/b"),
            (Action::Miss, "d.f"),
            (Action::Miss, "d.f"),
            (Action::Click0, "a/b"),
        ]);

        let res = cond_prob(
            &history,
            Action::Click1,
            FilterPred::new(UrlOrDom::Url("a/b")),
        );
        let expected = 2.0 / 4.0;
        assert_approx_eq!(f32, res, expected);

        let res = cond_prob(
            &history,
            Action::Skip,
            FilterPred::new(UrlOrDom::Url("a/b")),
        );
        let expected = 0. / 4.0;
        assert_approx_eq!(f32, res, expected);

        let res = cond_prob(
            &history,
            Action::Click1,
            FilterPred::new(UrlOrDom::Url("dodo")),
        );
        assert_approx_eq!(f32, res, 0.0);

        let res = cond_prob(
            &history,
            Action::Miss,
            FilterPred::new(UrlOrDom::Url("d.f")),
        );
        let expected = (2. + 1.) / (3. + 2.);
        assert_approx_eq!(f32, res, expected);

        let res = cond_prob(
            &history,
            Action::Miss,
            FilterPred::new(UrlOrDom::Url("dodo")),
        );
        assert_approx_eq!(f32, res, 0.0);
    }

    static HISTORY_FOR_URL: Lazy<Vec<HistSearchResult>> = Lazy::new(|| {
        history_by_url(&[
            /* query 0 */
            (Action::Skip, "1"),
            (Action::Click1, "2"),
            (Action::Skip, "3333"),
            (Action::Skip, "4"),
            (Action::Click1, "55"),
            (Action::Skip, "6"),
            (Action::Miss, "7"),
            (Action::Click2, "8"),
            (Action::Skip, "9"),
            (Action::Skip, "10"),
            /* query 1 */
            (Action::Click1, "1"),
            (Action::Click1, "2"),
            (Action::Click1, "4"),
            (Action::Click1, "3333"),
            (Action::Skip, "5"),
            (Action::Miss, "6"),
            (Action::Click1, "7"),
            (Action::Click1, "8"),
            (Action::Click1, "9"),
            (Action::Click1, "10"),
        ])
    });

    #[test]
    fn test_snippet_quality_for_url() {
        let current = &HISTORY_FOR_URL[4];
        let quality = snippet_quality(
            &HISTORY_FOR_URL,
            FilterPred::new(UrlOrDom::Url(&current.url)),
        );

        // 1 query matches
        //  first query
        //      no matching skip
        //      matching click
        //          clicked_documents = [2,55,8], current_document=55
        //          so index + 1 == 2
        //          score += 1/2
        //      score = 0.5
        //  total_score = 0.5
        //  nr_missed_or_skipped = 0
        //  => return 0
        assert_approx_eq!(f32, quality, 0.0);

        let current = &HISTORY_FOR_URL[2];
        let quality = snippet_quality(
            &HISTORY_FOR_URL,
            FilterPred::new(UrlOrDom::Url(&current.url)),
        );
        // 2 query matches
        //  first query
        //      matching skip
        //          clicked_documents = [2,55,8]
        //          score += -1/3
        //      no matching click
        //      score = -1/3
        //  second query
        //      no matching skip
        //      matching click
        //          clicked_documents = [1,2,4,3333,7,8,9,10], current_document=3333
        //          so index + 1 == 4
        //          score += 1/4
        //      score = 1/4
        //  total_score = -0.08333333333333331
        //  nr_missed_or_skipped = 1
        //  => return -0.08333333333333331 //closest f32 is -0.083_333_336
        assert_approx_eq!(f32, quality, -0.083_333_336);

        let current = &HISTORY_FOR_URL[7];
        let quality = snippet_quality(
            &HISTORY_FOR_URL,
            FilterPred::new(UrlOrDom::Url(&current.url)),
        );
        // 3 query matches
        //  first query
        //      no matching skip
        //      matching click
        //          clicked_documents = [2,55,8]
        //          score += -1/3
        //      score = -1/3
        //  second query
        //      no matching skip
        //      matching click
        //          clicked_documents = [1,2,4,3333,7,8,9,10], current_document=8
        //          so index + 1 == 6
        //          score += 1/6
        //      score = 1/6
        //  total_score = -0.16666666666666666
        //  nr_missed_or_skipped = 0
        //  => return 0
        assert_approx_eq!(f32, quality, 0.0);
    }

    fn history_by_domain<'a>(
        iter: impl IntoIterator<Item = &'a (Action, &'a str)>,
    ) -> Vec<HistSearchResult> {
        let mut queries = HashMap::new();

        iter.into_iter()
            .enumerate()
            .map(|(id, (action, domain))| {
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                HistSearchResult {
                    query: queries
                        .entry(per_query_id)
                        .or_insert_with(|| Query {
                            session_id: SessionId(mock_uuid(1)),
                            query_count: per_query_id,
                            query_id: QueryId(mock_uuid(per_query_id)),
                            query_words: vec![
                                "1".to_owned(),
                                "2".to_owned(),
                                per_query_id.to_string(),
                            ],
                        })
                        .clone(),
                    url: id.to_string(),
                    domain: (*domain).to_owned(),
                    final_rank: Rank(in_query_id),
                    day: DayOfWeek::Tue,
                    action: *action,
                }
            })
            .collect()
    }

    static HISTORY_FOR_DOMAIN: Lazy<Vec<HistSearchResult>> = Lazy::new(|| {
        history_by_domain(&[
            /* query 0 */
            (Action::Skip, "1"),
            (Action::Skip, "444"),
            (Action::Click1, "444"),
            (Action::Skip, "444"),
            (Action::Click1, "444"),
            (Action::Skip, "444"),
            (Action::Miss, "444"),
            (Action::Click2, "444"),
            (Action::Skip, "444"),
            (Action::Click1, "444"),
            /* query 1 */
            (Action::Click1, "444"),
            (Action::Click1, "444"),
            (Action::Click1, "444"),
            (Action::Click1, "444"),
            (Action::Skip, "12"),
            (Action::Miss, "444"),
            (Action::Click1, "444"),
            (Action::Click1, "444"),
            (Action::Click1, "9"),
            (Action::Click1, "10"),
            /* query 2 */
            (Action::Click1, "1"),
            (Action::Click1, "2"),
            (Action::Click1, "3"),
            (Action::Click1, "4"),
            (Action::Skip, "6"),
            (Action::Miss, "4"),
            (Action::Click1, "7"),
            (Action::Click1, "8"),
            (Action::Click1, "9"),
            (Action::Click1, "10"),
        ])
    });

    #[test]
    fn test_snippet_quality_for_domain() {
        let current = &HISTORY_FOR_DOMAIN[3];
        let quality = snippet_quality(
            &HISTORY_FOR_DOMAIN,
            FilterPred::new(UrlOrDom::Dom(&current.domain)),
        );
        // 2 query matches
        //  first query
        //      matching skip
        //          clicked_documents = [444,444,444,444]
        //          score += -1/4
        //      matching click
        //          clicked_documents = [444,444,....], current_document=444
        //          so index + 1 == 1
        //          score += 1/1
        //      score = 3/4
        //  second query
        //      no matching skip
        //      matching click
        //          clicked_documents = [444,444,...,9,10], current_document=444
        //          so index + 1 == 1
        //          score += 1/1
        //      score = 1
        //  total_score = 1.75
        //  nr_missed_or_skipped = 6
        //  => return 0.2916666666666667 // closest f32 is 0.291_666_66
        assert_approx_eq!(f32, quality, 0.291_666_66);

        let current = &HISTORY_FOR_DOMAIN[23];
        let quality = snippet_quality(
            &HISTORY_FOR_DOMAIN,
            FilterPred::new(UrlOrDom::Dom(&current.domain)),
        );
        // 1 query match
        //  first query
        //      no matching skip
        //      matching click
        //          clicked_documents = [1,2,3,4,7,8,9,10], current_document=4
        //          so index + 1 == 4
        //          score += 1/4
        //      score = 1/4
        //  total_score = 0.25
        //  nr_missed_or_skipped = 1
        //  return 0.25
        assert_approx_eq!(f32, quality, 0.25);
    }

    #[test]
    fn test_click_entropy() {
        let entropy = click_entropy(&HISTORY_FOR_DOMAIN[..]);
        assert_approx_eq!(f32, entropy, 3.108695);

        let entropy = click_entropy(&HISTORY_FOR_DOMAIN[20..]);
        assert_approx_eq!(f32, entropy, 3.0);

        let entropy = click_entropy(&HISTORY_FOR_DOMAIN[..15]);
        assert_approx_eq!(f32, entropy, 2.75);
    }

    fn history_by_day<'a>(
        iter: impl IntoIterator<Item = &'a (Action, DayOfWeek, &'a str)>,
    ) -> Vec<HistSearchResult> {
        let mut queries = HashMap::new();

        iter.into_iter()
            .enumerate()
            .map(|(id, (action, day, domain))| {
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                HistSearchResult {
                    query: queries
                        .entry(per_query_id)
                        .or_insert_with(|| Query {
                            session_id: SessionId(mock_uuid(1)),
                            query_count: per_query_id,
                            query_id: QueryId(mock_uuid(per_query_id)),
                            query_words: vec![
                                "1".to_owned(),
                                "2".to_owned(),
                                per_query_id.to_string(),
                            ],
                        })
                        .clone(),
                    url: id.to_string(),
                    domain: (*domain).to_owned(),
                    final_rank: Rank(in_query_id),
                    day: *day,
                    action: *action,
                }
            })
            .collect()
    }

    #[test]
    fn test_seasonality() {
        let history = history_by_day(&[
            (Action::Miss, DayOfWeek::Tue, "1"),
            (Action::Click1, DayOfWeek::Tue, "1"),
            (Action::Miss, DayOfWeek::Wed, "1"),
            (Action::Click1, DayOfWeek::Tue, "1"),
            (Action::Miss, DayOfWeek::Wed, "2"),
            (Action::Click1, DayOfWeek::Sun, "2"),
            (Action::Click1, DayOfWeek::Mon, "2"),
            (Action::Skip, DayOfWeek::Sun, "1"),
            (Action::Click1, DayOfWeek::Thu, "1"),
            (Action::Skip, DayOfWeek::Thu, "2"),
            (Action::Click1, DayOfWeek::Wed, "1"),
            (Action::Click1, DayOfWeek::Tue, "2"),
            (Action::Click1, DayOfWeek::Wed, "2"),
            (Action::Click1, DayOfWeek::Mon, "2"),
            (Action::Click1, DayOfWeek::Sat, "1"),
            (Action::Click1, DayOfWeek::Mon, "1"),
            (Action::Click1, DayOfWeek::Sat, "1"),
            (Action::Click1, DayOfWeek::Mon, "1"),
        ]);

        // seasonality = (5*(1+w_end_day))/(2*(1+working_day))
        // relevant thu/fr days: 1
        // relevant other days: 7
        let value = seasonality(&history, "1");
        assert_approx_eq!(f32, value, 0.625);

        // relevant thu/fr days: 0
        // relevant other days: 5
        let value = seasonality(&history, "2");
        assert_approx_eq!(f32, value, 0.416_666_66);

        assert_approx_eq!(f32, seasonality(&[], "1"), 0.0, ulps = 0);
    }

    #[test]
    fn test_terms_variety() {
        let query = Query {
            session_id: SessionId(mock_uuid(1244)),
            query_count: 1,
            query_id: QueryId(mock_uuid(12)),
            query_words: [3, 33, 12, 120, 33, 3]
                .iter()
                .map(|x| x.to_string())
                .collect(),
        };
        assert_eq!(terms_variety(&query), 4);
    }

    impl DayOfWeek {
        fn create_test_day(day: usize) -> Self {
            use DayOfWeek::*;
            match day % 7 {
                0 => Mon,
                1 => Tue,
                2 => Wed,
                3 => Thu,
                4 => Fri,
                5 => Sat,
                6 => Sun,
                _ => unreachable!(),
            }
        }
    }

    fn query_results_by_rank_and_relevance<'a>(
        iter: impl IntoIterator<Item = &'a (Rank, Action)>,
    ) -> Vec<HistSearchResult> {
        let mut queries = HashMap::new();

        iter.into_iter()
            .enumerate()
            .map(|(id, (rank, action))| {
                let per_query_id = id / 10;
                HistSearchResult {
                    query: queries
                        .entry(per_query_id)
                        .or_insert_with(|| Query {
                            session_id: SessionId(mock_uuid(1)),
                            query_count: per_query_id,
                            query_id: QueryId(mock_uuid(per_query_id)),
                            query_words: vec![per_query_id.to_string()],
                        })
                        .clone(),
                    url: id.to_string(),
                    domain: (id % 2).to_string(),
                    final_rank: *rank,
                    day: DayOfWeek::create_test_day(per_query_id),
                    action: *action,
                }
            })
            .collect()
    }

    #[test]
    fn test_mean_reciprocal_rank() {
        let history = query_results_by_rank_and_relevance(&[
            (Rank(0), Action::Skip),
            (Rank(1), Action::Miss),
            (Rank(2), Action::Click0),
            (Rank(3), Action::Click1),
            (Rank(4), Action::Click2),
        ]);

        let mrr = mean_recip_rank(&history, None, None);
        assert_approx_eq!(f32, mrr, 0.427_722_25);

        let mrr = mean_recip_rank(&history, None, Some(FilterPred::new(UrlOrDom::Dom("0"))));
        assert_approx_eq!(f32, mrr, 0.454_083_35);

        let mrr = mean_recip_rank(&history, Some(MrrOutcome::Miss), None);
        assert_approx_eq!(f32, mrr, 0.391_5);

        let mrr = mean_recip_rank(&history, Some(MrrOutcome::Skip), None);
        assert_approx_eq!(f32, mrr, 0.6415);

        let mrr = mean_recip_rank(&history, Some(MrrOutcome::Click), None);
        assert_approx_eq!(f32, mrr, 0.244_333_33);

        let mrr = mean_recip_rank(
            &history,
            Some(MrrOutcome::Click),
            Some(FilterPred::new(UrlOrDom::Dom("0"))),
        );
        assert_approx_eq!(f32, mrr, 0.241_5);
    }

    #[test]
    fn test_filter_predicate() {
        let mut result = HistSearchResult {
            query: Query {
                session_id: SessionId(mock_uuid(1)),
                query_count: 1,
                query_id: QueryId(mock_uuid(3)),
                query_words: vec!["ape".to_owned(), "tree".to_owned()],
            },
            url: "eleven".to_owned(),
            domain: "twenty/two".to_owned(),
            final_rank: Rank(0),
            day: DayOfWeek::Sun,
            action: Action::Click2,
        };

        let filter = FilterPred::new(UrlOrDom::Dom("42"));
        assert!(!filter.apply(&result));
        result.domain = "42".to_owned();
        assert!(filter.apply(&result));

        let filter = filter.with_query(QueryId(mock_uuid(25)));
        assert!(!filter.apply(&result));
        result.query.query_id = QueryId(mock_uuid(25));
        assert!(filter.apply(&result));

        let filter = filter.with_session(SessionCond::Current(SessionId(mock_uuid(100))));
        assert!(!filter.apply(&result));
        result.query.session_id = SessionId(mock_uuid(100));
        assert!(filter.apply(&result));

        let filter = filter.with_session(SessionCond::NotCurrent {
            current: SessionId(mock_uuid(100)),
        });
        assert!(!filter.apply(&result));
        result.query.session_id = SessionId(mock_uuid(80));
        assert!(filter.apply(&result));

        let filter = filter.with_session(SessionCond::All);
        result.query.session_id = SessionId(mock_uuid(3333));
        assert!(filter.apply(&result));

        let filter = FilterPred::new(UrlOrDom::Url("42"));
        assert!(!filter.apply(&result));
        result.url = "42".to_owned();
        assert!(filter.apply(&result));
    }

    #[test]
    fn the_right_aggregate_atoms_are_chosen() {
        use UrlOrDom::*;

        let click_mrr = AtomFeat::MeanRecipRank(MrrOutcome::Click);
        let miss_mrr = AtomFeat::MeanRecipRank(MrrOutcome::Miss);
        let skip_mrr = AtomFeat::MeanRecipRank(MrrOutcome::Skip);
        let combine_mrr = AtomFeat::MeanRecipRankAll;
        let click2 = AtomFeat::CondProb(Action::Click2);
        let missed = AtomFeat::CondProb(Action::Miss);
        let skipped = AtomFeat::CondProb(Action::Skip);
        let snipped_quality = AtomFeat::SnippetQuality;

        // No rust formatting for readability.
        // This is formatted as a table, which rustfmt would break.
        #[rustfmt::skip]
        let test_cases = vec![
            /* url.usual */
            (Url("1"),    None,       SessionCond::All, vec![click_mrr, click2, missed, snipped_quality]),
            /* url.anterior */
            (Url("1"),    None,       not_current(3),   vec![click2, missed, snipped_quality]),
            /* url.session (not used) */
            (Url("1"),    None,       current(3),       vec![]),
            /* url.query */
            (Url("1"),    Some(32),   SessionCond::All, vec![combine_mrr, click2, missed, snipped_quality]),
            /* url.query_anterior */
            (Url("1"),    Some(32),   not_current(3),   vec![combine_mrr, click_mrr, miss_mrr, skip_mrr, missed, snipped_quality]),
            /* url.query_session */
            (Url("1"),    Some(32),   current(3),       vec![miss_mrr]),
            /* domain.usual */
            (Dom("1"),    None,       SessionCond::All, vec![skipped, missed, click2, snipped_quality]),
            /* domain.anterior */
            (Dom("1"),    None,       not_current(3),   vec![click2, missed, snipped_quality]),
            /* domain.session (not used) */
            (Dom("1"),    None,       current(3),       vec![]),
            /* domain.query */
            (Dom("1"),    Some(32),   SessionCond::All, vec![missed, snipped_quality, miss_mrr]),
            /* domain.anterior */
            (Dom("1"),    Some(32),   not_current(3),   vec![snipped_quality]),
            /* domain.query_session (not used) */
            (Dom("1"),    Some(32),   current(3),       vec![]),
        ];

        for (url_or_dom, query_filter, session_cond, expected) in test_cases.into_iter() {
            let mut filter = FilterPred::new(url_or_dom).with_session(session_cond);
            if let Some(query) = query_filter {
                filter = filter.with_query(QueryId(mock_uuid(query)));
            }

            let agg_atoms = filter.agg_atoms();
            if agg_atoms.len() != expected.len()
                || !expected.iter().all(|atom| agg_atoms.contains(atom))
            {
                panic!(
                    "for ({:?}, {:?}, {:?}) expected {:?} but got {:?}",
                    url_or_dom, query_filter, session_cond, expected, agg_atoms
                );
            }
        }

        //----
        fn not_current(sub_id: usize) -> SessionCond {
            SessionCond::NotCurrent {
                current: SessionId(mock_uuid(sub_id)),
            }
        }

        fn current(sub_id: usize) -> SessionCond {
            SessionCond::Current(SessionId(mock_uuid(sub_id)))
        }
    }

    #[test]
    fn test_is_clicked() {
        let mut result = HistSearchResult {
            query: Query {
                session_id: SessionId(mock_uuid(1)),
                query_count: 1,
                query_id: QueryId(mock_uuid(3)),
                query_words: vec!["ape".to_owned(), "tree".to_owned()],
            },
            url: "eleven".to_owned(),
            domain: "twenty/two".to_owned(),
            final_rank: Rank(0),
            day: DayOfWeek::Sun,
            action: Action::Skip,
        };
        assert!(!result.is_clicked());

        result.action = Action::Miss;
        assert!(!result.is_clicked());

        result.action = Action::Click0;
        assert!(!result.is_clicked());

        result.action = Action::Click1;
        assert!(result.is_clicked());

        result.action = Action::Click2;
        assert!(result.is_clicked());
    }

    //TODO turn this into end to end tests where you compare the extracted feature arrays (once we do create them)
    //rustfmt makes all the assert_approx_eq much less readable in this specific case.
    #[rustfmt::skip]
    fn do_test_compute_features(history: Vec<HistSearchResult>, search_results: Vec<DocSearchResult>, expected_features: &[[f32;50]]) {
        let got_features = build_features(history, search_results).unwrap();

        let click_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Click);
        let miss_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Miss);
        let skip_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Skip);
        let combine_mrr = &AtomFeat::MeanRecipRankAll;
        let click2 = &AtomFeat::CondProb(Action::Click2);
        let missed = &AtomFeat::CondProb(Action::Miss);
        let skipped = &AtomFeat::CondProb(Action::Skip);
        let snippet_quality = &AtomFeat::SnippetQuality;

        for (features, feature_row) in izip!(got_features, expected_features) {
            let Features {
                initial_rank,
                aggregated,
                user,
                query,
                cumulated,
                terms_variety,
                seasonality
            } = features;

            assert_approx_eq!(f32, feature_row[0], f32::from(initial_rank), ulps = 0);

            assert_approx_eq!(f32, feature_row[1], aggregated.url[click_mrr]);
            assert_approx_eq!(f32, feature_row[2], aggregated.url[click2]);
            assert_approx_eq!(f32, feature_row[3], aggregated.url[missed]);
            assert_approx_eq!(f32, feature_row[4], aggregated.url[snippet_quality]);
            assert_approx_eq!(f32, feature_row[5], aggregated.url_ant[click2]);
            assert_approx_eq!(f32, feature_row[6], aggregated.url_ant[missed]);
            assert_approx_eq!(f32, feature_row[7], aggregated.url_ant[snippet_quality]);
            assert_approx_eq!(f32, feature_row[8], aggregated.url_query[combine_mrr]);
            assert_approx_eq!(f32, feature_row[9], aggregated.url_query[click2]);
            assert_approx_eq!(f32, feature_row[10], aggregated.url_query[missed]);
            assert_approx_eq!(f32, feature_row[11], aggregated.url_query[snippet_quality]);
            assert_approx_eq!(f32, feature_row[12], aggregated.url_query_ant[combine_mrr]);
            assert_approx_eq!(f32, feature_row[13], aggregated.url_query_ant[click_mrr]);
            assert_approx_eq!(f32, feature_row[14], aggregated.url_query_ant[miss_mrr]);
            assert_approx_eq!(f32, feature_row[15], aggregated.url_query_ant[skip_mrr]);
            assert_approx_eq!(f32, feature_row[16], aggregated.url_query_ant[missed]);
            assert_approx_eq!(f32, feature_row[17], aggregated.url_query_ant[snippet_quality]);
            assert_approx_eq!(f32, feature_row[18], aggregated.url_query_curr[miss_mrr]);
            assert_approx_eq!(f32, feature_row[19], aggregated.dom[skipped]);
            assert_approx_eq!(f32, feature_row[20], aggregated.dom[missed]);
            assert_approx_eq!(f32, feature_row[21], aggregated.dom[click2]);
            assert_approx_eq!(f32, feature_row[22], aggregated.dom[snippet_quality]);
            assert_approx_eq!(f32, feature_row[23], aggregated.dom_ant[click2]);
            assert_approx_eq!(f32, feature_row[24], aggregated.dom_ant[missed]);
            assert_approx_eq!(f32, feature_row[25], aggregated.dom_ant[snippet_quality]);
            assert_approx_eq!(f32, feature_row[26], aggregated.dom_query[missed]);
            assert_approx_eq!(f32, feature_row[27], aggregated.dom_query[snippet_quality]);
            assert_approx_eq!(f32, feature_row[28], aggregated.dom_query[miss_mrr]);
            assert_approx_eq!(f32, feature_row[29], aggregated.dom_query_ant[snippet_quality]);

            let QueryFeatures {
                click_entropy,
                num_terms,
                mean_query_count,
                mean_occurs_per_session,
                num_occurs,
                click_mrr,
                mean_clicks,
                mean_non_clicks
            } = query;

            assert_approx_eq!(f32, feature_row[30], click_entropy);
            assert_approx_eq!(f32, feature_row[31], num_terms as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[32], mean_query_count);
            assert_approx_eq!(f32, feature_row[33], mean_occurs_per_session);
            assert_approx_eq!(f32, feature_row[34], num_occurs as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[35], click_mrr);
            assert_approx_eq!(f32, feature_row[36], mean_clicks);
            assert_approx_eq!(f32, feature_row[37], mean_non_clicks);

            let UserFeatures {
                click_entropy,
                click_counts,
                num_queries,
                mean_words_per_query,
                mean_unique_words_per_session,
            } = user;
            assert_approx_eq!(f32, feature_row[38], click_entropy);
            assert_approx_eq!(f32, feature_row[39], click_counts.click12 as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[40], click_counts.click345 as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[41], click_counts.click6up as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[42], num_queries as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[43], mean_words_per_query);
            assert_approx_eq!(f32, feature_row[44], mean_unique_words_per_session);

            assert_approx_eq!(f32, feature_row[45], cumulated.url_skip);
            assert_approx_eq!(f32, feature_row[46], cumulated.url_click1);
            assert_approx_eq!(f32, feature_row[47], cumulated.url_click2);

            assert_approx_eq!(f32, feature_row[48], terms_variety as f32, ulps = 0);

            assert_approx_eq!(f32, feature_row[49], seasonality);
        }
    }

    #[test]
    fn test_full_feature_extraction_simple() {
        //auto generated
        #[rustfmt::skip]
        let history = vec![
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "26142648".to_owned(), domain: "2597528".to_owned(), action: Action::Click2, final_rank: Rank(0), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "44200215".to_owned(), domain: "3852697".to_owned(), action: Action::Miss, final_rank: Rank(1), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "40218620".to_owned(), domain: "3602893".to_owned(), action: Action::Miss, final_rank: Rank(2), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "21854374".to_owned(), domain: "2247911".to_owned(), action: Action::Miss, final_rank: Rank(3), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "6152223".to_owned(), domain: "787424".to_owned(), action: Action::Miss, final_rank: Rank(4), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "46396840".to_owned(), domain: "3965502".to_owned(), action: Action::Miss, final_rank: Rank(5), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "65705884".to_owned(), domain: "4978404".to_owned(), action: Action::Miss, final_rank: Rank(6), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "4607041".to_owned(), domain: "608358".to_owned(), action: Action::Miss, final_rank: Rank(7), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "60306140".to_owned(), domain: "4679885".to_owned(), action: Action::Miss, final_rank: Rank(8), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(12283852)), query_words: vec!["3468976".to_owned(), "4614115".to_owned()], query_count: 0, }, day: DayOfWeek::Fri, url: "1991065".to_owned(), domain: "295576".to_owned(), action: Action::Miss, final_rank: Rank(9), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "43220173".to_owned(), domain: "3802280".to_owned(), action: Action::Click2, final_rank: Rank(0),  },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "68391867".to_owned(), domain: "5124172".to_owned(), action: Action::Miss, final_rank: Rank(1), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "48241082".to_owned(), domain: "4077775".to_owned(), action: Action::Miss, final_rank: Rank(2), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "28461283".to_owned(), domain: "2809381".to_owned(), action: Action::Miss, final_rank: Rank(3), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "36214392".to_owned(), domain: "3398386".to_owned(), action: Action::Miss, final_rank: Rank(4), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "26215090".to_owned(), domain: "2597528".to_owned(), action: Action::Miss, final_rank: Rank(5), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "55157032".to_owned(), domain: "4429726".to_owned(), action: Action::Miss, final_rank: Rank(6), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "35921251".to_owned(), domain: "3380635".to_owned(), action: Action::Miss, final_rank: Rank(7), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "37498049".to_owned(), domain: "3463275".to_owned(), action: Action::Miss, final_rank: Rank(8), },
            HistSearchResult { query: Query { session_id: SessionId(mock_uuid(2746324)),query_id: QueryId(mock_uuid(7297472)), query_words: vec!["2758230".to_owned()], query_count: 1, }, day: DayOfWeek::Fri, url: "70173304".to_owned(), domain: "5167485".to_owned(), action: Action::Miss, final_rank: Rank(9), },
        ];

        //auto generated
        #[rustfmt::skip]
        let current_search_results = vec![
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "41131641".to_owned(), domain: "3661944".to_owned(), initial_rank: Rank(0), },
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "43630521".to_owned(), domain: "3823198".to_owned(), initial_rank: Rank(1), },
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "28819788".to_owned(), domain: "2832997".to_owned(), initial_rank: Rank(2), },
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "28630417".to_owned(), domain: "2819308".to_owned(), initial_rank: Rank(3), },
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "49489872".to_owned(), domain: "4155543".to_owned(), initial_rank: Rank(4), },
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "1819187".to_owned(), domain: "269174".to_owned(), initial_rank: Rank(5), },
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "27680026".to_owned(), domain: "2696111".to_owned(), initial_rank: Rank(6), },
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "1317174".to_owned(), domain: "207936".to_owned(), initial_rank: Rank(7), },
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "28324834".to_owned(), domain: "2790971".to_owned(), initial_rank: Rank(8), },
            DocSearchResult { query: Query { session_id: SessionId(mock_uuid(2746325)), query_id: QueryId(mock_uuid(20331734)), query_words: vec!["4631619".to_owned(), "2289501".to_owned()], query_count: 0, }, url: "54208271".to_owned(), domain: "4389621".to_owned(), initial_rank: Rank(9), },
        ];

        //auto generated
        #[rustfmt::skip]
        let features = &[
            [1.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [2.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [3.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [4.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [5.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [6.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [7.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [8.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [9.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [10.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0]
        ];

        do_test_compute_features(history, current_search_results, features);
    }

    const TEST_DATA_DIR: &str = "../data/ltr_feature_extraction_tests_v0000";
    const TEST_HISTORY_FILE_NAME: &str = "history.csv";
    const TEST_CURRENT_QUERY_FILE_NAME: &str = "current_query.csv";
    const TEST_FEATURES_FILE_NAME: &str = "features.csv";

    #[derive(Deserialize)]
    enum SoundgardenRelevance {
        High,
        Medium,
        Miss,
        Skip,
    }

    impl From<SoundgardenRelevance> for Action {
        fn from(relevance: SoundgardenRelevance) -> Action {
            use SoundgardenRelevance::*;

            match relevance {
                High => Action::Click2,
                Medium => Action::Click1,
                Miss => Action::Miss,
                Skip => Action::Skip,
            }
        }
    }

    #[derive(Deserialize)]
    struct SoundgardenHistoryCsvHelper {
        session_id: usize,
        #[allow(dead_code)]
        user_id: usize,
        query_id: usize,
        day: usize,
        query_words: String,
        url: String,
        domain: String,
        relevance: SoundgardenRelevance,
        position: usize,
        query_counter: usize,
    }

    impl From<SoundgardenHistoryCsvHelper> for HistSearchResult {
        fn from(csv: SoundgardenHistoryCsvHelper) -> HistSearchResult {
            HistSearchResult {
                query: Query {
                    session_id: SessionId(mock_uuid(csv.session_id)),
                    query_count: csv.query_counter,
                    query_id: QueryId(mock_uuid(csv.query_id)),
                    query_words: csv.query_words.split(',').map(Into::into).collect(),
                },
                url: csv.url,
                domain: csv.domain,
                final_rank: Rank(csv.position.checked_sub(1).unwrap()),
                day: DayOfWeek::create_test_day(csv.day),
                action: csv.relevance.into(),
            }
        }
    }

    fn read_test_history(path: impl AsRef<Path>) -> Vec<HistSearchResult> {
        let mut reader = csv::Reader::from_path(path).unwrap();
        reader
            .deserialize()
            .map(|record: Result<SoundgardenHistoryCsvHelper, _>| record.unwrap().into())
            .collect()
    }

    #[derive(Deserialize)]
    struct SoundgardenDocumentsCsvHelper {
        session_id: usize,
        #[allow(dead_code)]
        user_id: usize,
        query_id: usize,
        #[allow(dead_code)]
        day: usize,
        query_words: String,
        url: String,
        domain: String,
        initial_rank: usize,
        query_counter: usize,
    }

    impl From<SoundgardenDocumentsCsvHelper> for DocSearchResult {
        fn from(csv: SoundgardenDocumentsCsvHelper) -> Self {
            DocSearchResult {
                query: Query {
                    session_id: SessionId(mock_uuid(csv.session_id)),
                    query_count: csv.query_counter,
                    query_id: QueryId(mock_uuid(csv.query_id)),
                    query_words: csv.query_words.split(',').map(Into::into).collect(),
                },
                url: csv.url,
                domain: csv.domain,
                initial_rank: Rank(csv.initial_rank.checked_sub(1).unwrap()),
            }
        }
    }

    fn read_test_query(path: impl AsRef<Path>) -> Vec<DocSearchResult> {
        let mut reader = csv::Reader::from_path(path).unwrap();
        reader
            .deserialize()
            .map(|record: Result<SoundgardenDocumentsCsvHelper, _>| record.unwrap().into())
            .collect()
    }

    fn read_test_features(path: impl AsRef<Path>) -> Vec<[f32; 50]> {
        let mut reader = csv::Reader::from_path(path).unwrap();
        // We have a header but don't want to use it for deserialization,
        // because of this we use `StringRecord::deserialize(None)` instead of
        // `Reader::deserialize()`.
        reader
            .records()
            .map(|record| {
                let row: Vec<_> = record.unwrap().deserialize(None).unwrap();
                <[f32; 50]>::try_from(row.as_slice()).unwrap()
            })
            .collect()
    }

    #[test]
    fn test_full_feature_extraction() {
        let mut did_run = false;
        for test in Path::new(TEST_DATA_DIR)
            .read_dir()
            .unwrap()
            .map(|d| d.unwrap().path())
        {
            did_run = true;

            if test.ends_with("sha256sums") {
                continue;
            }

            let history = read_test_history(test.join(TEST_HISTORY_FILE_NAME));
            let current_search_results = read_test_query(test.join(TEST_CURRENT_QUERY_FILE_NAME));
            let features = read_test_features(test.join(TEST_FEATURES_FILE_NAME));

            do_test_compute_features(history, current_search_results, &features);
        }

        if !did_run {
            panic!("Missing test cases. Did you run ./download_data.sh.");
        }
    }
}
