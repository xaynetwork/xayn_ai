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
            SessionCond::Anterior(id) => session_id != id,
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
    aggregate: AggregateFeatures,
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
        let seasonality = seasonality(hists, doc);

        Self {
            initial_rank,
            aggregate,
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
            aggregate,
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
fn seasonality(hists: &[HistSearchResult], doc: &DocSearchResult) -> f32 {
    let (clicks_wknd, clicks_wkday) = hists
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
