//! Feature extraction algorithm based on Dataiku's solution to Yandex Personalized Web Search
//! Challenge [1]. See [2] for the first implementation in Python.
//!
//! [1]: https://www.academia.edu/9872579/Dataikus_Solution_to_Yandexs_Personalized_Web_Search_Challenge
//! [2]: https://github.com/xaynetwork/soundgarden

// FIXME[comment/resolve with review]: Terminology
// - SearchResult: Has a unclear name due to its overlap with `std::result::Result` and the fact
//   that it's not clear that this is part of the search query result not the full query result.
//   We also use the term Document for it in other places.
// - ResultSet: While it happens to be a set it mainly is a list ordered by ranking (ascending).
// - rank vs. position
// - ClickSat vs. Relevance
//
// FIXME[comment/resolve maybe in other task]: ClickSat::Low
// - What does ClickSat::Low mean? We treat it like `Miss` in some
//   places but not like `Miss` in other places. Sometimes we differ
//   in it's treatment in the same function... (Like not counting it
//   in the miss_or_skip_count, but also not counting it as clicked.)
//
// FIXME[comment/resolve maybe in other task]: ClickSat::Low
// - Why is there a query_id field and query_counter field?
//   Does it still matter for us or did it only matter for training?
// -
#![allow(dead_code)] // TEMP

use itertools::Itertools;
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;

/// Click satisfaction score.
///
/// Based on Yandex notion of dwell-time: time elapsed between a click and the next action.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
#[cfg_attr(test, derive(Debug))]
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
#[cfg_attr(test, derive(Copy, Clone))]
pub(crate) enum DayOfWeek {
    Mon,
    Tue,
    Wed,
    Thu,
    Fri,
    Sat,
    Sun,
}

/// A single search query result before being reranked
pub struct CurrentSearchResult {
    /// Session identifier.
    pub(crate) session_id: i32,
    /// User identifier.
    pub(crate) user_id: i32,
    /// Query identifier.
    pub(crate) query_id: i32,
    /// Day of week search was performed.
    pub(crate) day: DayOfWeek,
    /// Words of the query, each masked.
    pub(crate) query_words: Vec<i32>,
    /// URL of result, masked.
    pub(crate) url: i32,
    /// Domain of result, masked.
    pub(crate) domain: i32,
    /// Position among other results.
    pub(crate) initial_rank: Rank,
    /// Query count within session.
    pub(crate) query_counter: u8,
}

/// Data pertaining to a single result from a search.
#[cfg_attr(test, derive(Clone))]
pub struct SearchResult {
    /// Session identifier.
    pub(crate) session_id: i32,
    /// User identifier.
    pub(crate) user_id: i32,
    /// Query identifier.
    pub(crate) query_id: i32,
    /// Day of week search was performed.
    pub(crate) day: DayOfWeek,
    /// Words of the query, each masked.
    pub(crate) query_words: Vec<i32>,
    /// URL of result, masked.
    pub(crate) url: i32,
    /// Domain of result, masked.
    pub(crate) domain: i32,
    /// Relevance level of the result.
    pub(crate) relevance: ClickSat,
    /// Position among other results.
    pub(crate) position: Rank,
    /// Query count within session.
    pub(crate) query_counter: u8,
}

impl SearchResult {
    pub(crate) fn is_clicked(&self) -> bool {
        //Note: How do we treat `Low`?
        self.relevance > ClickSat::Low
    }

    pub(crate) fn is_skipped(&self) -> bool {
        self.relevance == ClickSat::Skip
    }
}

impl AsRef<SearchResult> for SearchResult {
    fn as_ref(&self) -> &SearchResult {
        self
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
    // FIXME[comment/resolve with review]: (aboves todo)
    //      We need the list of documents/urls+metadata returned by the search query in
    //      ascending ranking order. As such the only way to relax this is by manually
    //      sorting the search query result.
    fn new(rs: Vec<&'a SearchResult>) -> Self {
        Self(rs)
    }

    /// Iterates over all documents by in ascending ranking order.
    fn documents(&self) -> impl Iterator<Item = &'a SearchResult> + '_ {
        self.0.iter().copied()
    }

    /// Iterates over all documents which have been clicked in ascending ranking order.
    fn clicked_documents(&self) -> impl Iterator<Item = &'a SearchResult> + '_ {
        self.0.iter().flat_map(|doc| doc.is_clicked().then(|| *doc))
    }
}

pub(crate) struct Query {
    pub(crate) id: i32,
    pub(crate) words: Vec<i32>,
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
#[cfg_attr(test, derive(Debug))]
pub(crate) enum UrlOrDom {
    /// A specific URL.
    Url(i32),
    /// Any URL belonging to the given domain.
    Dom(i32),
}

/// Query submission timescale.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug))]
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
pub(crate) struct FilterPred {
    doc: UrlOrDom,
    query: Option<i32>,
    session: SessionCond,
}

impl FilterPred {
    pub(crate) fn new(doc: UrlOrDom) -> Self {
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
            (Url(_), Some(_), Ant(_)) => smallvec![mrr, MRR(Click), MRR(Miss), MRR(Skip), miss, SQ],
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
///
/// # Implementation Differences
///
/// Soundgarden uses the "query_array" instead of the current session
/// (which beside the current query is in history).
///
/// While this is clearly a bug we need to keep it and use the
/// current results. In turn we don't need to filter for the
/// session id as all current results are from
/// the current search query and as such the current session.
///
/// Additionally this also causes all inspected results/documents
/// to have the same query words and in turn this is just number
/// of unique words in the current query, which is kinda very pointless.
fn terms_variety(current_query: &Query) -> usize {
    current_query.words.iter().unique().count()
}

/// Weekend seasonality of a given domain.
///
/// If there are no matching entries for the given domain `0`
/// is returned, even through normally you would expect 2.5
/// to be returned. But we need to keep it in sync with the
/// python code.
fn seasonality(history: &[SearchResult], domain: i32) -> f32 {
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

    if clicks_wknd + clicks_wkday == 0 {
        0.0
    } else {
        2.5 * (1. + clicks_wknd as f32) / (1. + clicks_wkday as f32)
    }
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
///       sum{score(query)}
/// --------------------------
/// |hist({Miss, Skip}, pred)|
/// ```
///
/// where the sum ranges over all query results containing a result `r` with URL/Domain matching the URL/Domain of `res`.
///
/// If `|hist({Miss, Skip}, pred)|` is `0` then `0` is returned.
pub(crate) fn snippet_quality(hist: &[SearchResult], pred: FilterPred) -> f32 {
    let miss_or_skip_count = hist
        .iter()
        .filter(|r| {
            pred.apply(r) && (r.relevance == ClickSat::Miss || r.relevance == ClickSat::Skip)
        })
        .count() as f32;

    if miss_or_skip_count == 0. {
        return 0.;
    }

    let total_score = hist
        .iter()
        .group_by(|r| (r.session_id, r.query_counter))
        .into_iter()
        .filter_map(|(_, res)| {
            let res = ResultSet::new(res.into_iter().collect());
            let has_match = res.documents().any(|doc| pred.apply(doc));
            has_match.then(|| snippet_score(&res, pred))
        })
        .sum::<f32>();

    total_score / miss_or_skip_count
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
/// The python implementation on which basis the model was trained does not match the
/// definition.
///
/// For every `outcome` except `ClickSat::Miss` following formula is used:
///
/// ```text
/// |hist(outcome, pred)|
/// ----------------------
///    |hist(pred)|
/// ```
///
/// If the `outcome` is `ClickSet::Miss` then following formula is used instead:
///
/// ```text
///       |hist(outcome, pred)| + 1
/// ---------------------------------------
/// |hist(outcome, pred)| + |hist(pred)|
/// ```
///
/// In both cases it defaults to 0 if the denominator is 0 (as in this case the
/// numerator should be 0 too)
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

#[cfg(test)]
mod tests {
    use itertools::izip;
    use once_cell::sync::Lazy;

    /* Pseudo Code based on the Python Implementation

    FIXME remove before merging but not before review.

    ## Query Quality Pseudo Code

    ```pseudo
    considered_data['all'] == history
    action_predicate_map['all'] == considered_data['all'][where matches current url/domain]
                                == history[where matches current url/domain]
    action_predicate_map['missed'] == history[where relevance==missed and matches current url/domain]
    action_predicate_map['skipped'] == history[where relevance==skipped and matches current url/domain]

    matching_queries == all queries which have at least one matching query

    scores = sum(score(query,...) for query in queries)

    scores / len(action_predicate_map['missed'])+len(action_prediacte_map['skipped'])

    ON divide by 0 return 0
    ```

    ## Query Score Pseudo Code

    ```pseudo
    query_matched_documents == query[where matches current url/domain]
                            at least length 1 or we would not have called
                            this function

    clicked_documents == query[where matches current url/domain and relevance > 0].url/domain

    there_is_a_skip = skipped_results_match.shape[0] > 0 == true iff there is at least one item in query_matched_documents which is also in action_predicate_map['skipped']

    if there_is_a_skip:
        score += -1/clicked_documents.size

    there_is_a_match = clicked_results_match.shape[0] > 0 == true iff there is at least one item in query_matched_documents which is also in action_predicate_map['clicked']

    if there_is_a_match:
        // all indices of clicked_documents which match the current url/domain
        click_position_index = indices where [clicked_documents == current_document] is true
        score += 1/(click_position_index[0] + 1)


    ON /0 in scores += <term> just don't change it
    */

    use crate::ltr::features::{
        aggregate::AggregateFeatures,
        cumulate::CumFeatures,
        query::QueryFeatures,
        user::UserFeatures,
    };

    use super::*;

    impl Rank {
        pub(crate) fn from_usize(rank: usize) -> Self {
            match rank {
                1 => Rank::First,
                2 => Rank::Second,
                3 => Rank::Third,
                4 => Rank::Fourth,
                5 => Rank::Fifth,
                6 => Rank::Sixth,
                7 => Rank::Seventh,
                8 => Rank::Eighth,
                9 => Rank::Nineth,
                10 => Rank::Last,
                _ => panic!("Only support rank 1-10, got {}", rank),
            }
        }
    }

    // helper to reduce some repetition in tests
    impl From<SearchResult> for CurrentSearchResult {
        fn from(res: SearchResult) -> Self {
            let SearchResult {
                session_id,
                user_id,
                query_id,
                day,
                query_words,
                url,
                domain,
                relevance: _,
                position,
                query_counter,
            } = res;
            CurrentSearchResult {
                session_id,
                user_id,
                query_id,
                day,
                query_words,
                url,
                domain,
                initial_rank: position,
                query_counter,
            }
        }
    }

    fn history_by_url<'a>(
        iter: impl IntoIterator<Item = &'a (ClickSat, i32)>,
    ) -> Vec<SearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (relevance, url))| {
                let id = id as i32;
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                SearchResult {
                    session_id: 1,
                    user_id: 1,
                    query_id: per_query_id,
                    day: DayOfWeek::Tue,
                    query_words: vec![1, 2, id],
                    url: *url,
                    domain: 42,
                    relevance: *relevance,
                    position: Rank::from_usize(1 + in_query_id as usize),
                    query_counter: per_query_id as u8,
                }
            })
            .collect()
    }

    fn history_by_domain<'a>(
        iter: impl IntoIterator<Item = &'a (ClickSat, i32)>,
    ) -> Vec<SearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (relevance, domain))| {
                let id = id as i32;
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                SearchResult {
                    session_id: 1,
                    user_id: 1,
                    query_id: per_query_id,
                    day: DayOfWeek::Tue,
                    query_words: vec![1, 2, id],
                    url: id,
                    domain: *domain,
                    relevance: *relevance,
                    position: Rank::from_usize(1 + in_query_id as usize),
                    query_counter: per_query_id as u8,
                }
            })
            .collect()
    }

    #[test]
    fn test_cond_prob() {
        let history = history_by_url(&[
            (ClickSat::Medium, 10),
            (ClickSat::High, 10),
            (ClickSat::Medium, 12),
            (ClickSat::Medium, 8),
            (ClickSat::Medium, 10),
            (ClickSat::Miss, 12),
            (ClickSat::Miss, 12),
            (ClickSat::Low, 10),
        ]);

        let res = cond_prob(
            &history,
            ClickSat::Medium,
            FilterPred::new(UrlOrDom::Url(10)),
        );
        let expected = 2.0 / 4.0;
        assert_approx_eq!(f32, res, expected);

        let res = cond_prob(&history, ClickSat::Skip, FilterPred::new(UrlOrDom::Url(10)));
        let expected = 0. / 4.0;
        assert_approx_eq!(f32, res, expected);

        let res = cond_prob(
            &history,
            ClickSat::Medium,
            FilterPred::new(UrlOrDom::Url(100)),
        );
        assert_approx_eq!(f32, res, 0.0);

        let res = cond_prob(&history, ClickSat::Miss, FilterPred::new(UrlOrDom::Url(12)));
        let expected = (2. + 1.) / (3. + 2.);
        assert_approx_eq!(f32, res, expected);

        let res = cond_prob(
            &history,
            ClickSat::Miss,
            FilterPred::new(UrlOrDom::Url(100)),
        );
        assert_approx_eq!(f32, res, 0.0);
    }

    static HISTORY_FOR_URL: Lazy<Vec<SearchResult>> = Lazy::new(|| {
        history_by_url(&[
            /* query 0 */
            (ClickSat::Skip, 1),
            (ClickSat::Medium, 2),
            (ClickSat::Skip, 3333),
            (ClickSat::Skip, 4),
            (ClickSat::Medium, 55),
            (ClickSat::Skip, 6),
            (ClickSat::Miss, 7),
            (ClickSat::High, 8),
            (ClickSat::Skip, 9),
            (ClickSat::Skip, 10),
            /* query 1 */
            (ClickSat::Medium, 1),
            (ClickSat::Medium, 2),
            (ClickSat::Medium, 4),
            (ClickSat::Medium, 3333),
            (ClickSat::Skip, 5),
            (ClickSat::Miss, 6),
            (ClickSat::Medium, 7),
            (ClickSat::Medium, 8),
            (ClickSat::Medium, 9),
            (ClickSat::Medium, 10),
        ])
    });

    #[test]
    fn test_snippet_quality_for_url() {
        let current = &HISTORY_FOR_URL[4];
        let quality = snippet_quality(
            &HISTORY_FOR_URL,
            FilterPred::new(UrlOrDom::Url(current.url)),
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
            FilterPred::new(UrlOrDom::Url(current.url)),
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
            FilterPred::new(UrlOrDom::Url(current.url)),
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

    static HISTORY_FOR_DOMAIN: Lazy<Vec<SearchResult>> = Lazy::new(|| {
        history_by_domain(&[
            /* query 0 */
            (ClickSat::Skip, 1),
            (ClickSat::Skip, 444),
            (ClickSat::Medium, 444),
            (ClickSat::Skip, 444),
            (ClickSat::Medium, 444),
            (ClickSat::Skip, 444),
            (ClickSat::Miss, 444),
            (ClickSat::High, 444),
            (ClickSat::Skip, 444),
            (ClickSat::Medium, 444),
            /* query 1 */
            (ClickSat::Medium, 444),
            (ClickSat::Medium, 444),
            (ClickSat::Medium, 444),
            (ClickSat::Medium, 444),
            (ClickSat::Skip, 12),
            (ClickSat::Miss, 444),
            (ClickSat::Medium, 444),
            (ClickSat::Medium, 444),
            (ClickSat::Medium, 9),
            (ClickSat::Medium, 10),
            /* query 2 */
            (ClickSat::Medium, 1),
            (ClickSat::Medium, 2),
            (ClickSat::Medium, 3),
            (ClickSat::Medium, 4),
            (ClickSat::Skip, 6),
            (ClickSat::Miss, 4),
            (ClickSat::Medium, 7),
            (ClickSat::Medium, 8),
            (ClickSat::Medium, 9),
            (ClickSat::Medium, 10),
        ])
    });

    #[test]
    fn test_snippet_quality_for_domain() {
        let current = &HISTORY_FOR_DOMAIN[3];
        let quality = snippet_quality(
            &HISTORY_FOR_DOMAIN,
            FilterPred::new(UrlOrDom::Dom(current.domain)),
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
            FilterPred::new(UrlOrDom::Dom(current.domain)),
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
        iter: impl IntoIterator<Item = &'a (ClickSat, DayOfWeek, i32)>,
    ) -> Vec<SearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (relevance, day, domain))| {
                let id = id as i32;
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                SearchResult {
                    session_id: 1,
                    user_id: 1,
                    query_id: per_query_id,
                    day: *day,
                    query_words: vec![1, 2, id],
                    url: id,
                    domain: *domain,
                    relevance: *relevance,
                    position: Rank::from_usize(1 + in_query_id as usize),
                    query_counter: per_query_id as u8,
                }
            })
            .collect()
    }

    #[test]
    fn test_seasonality() {
        let history = history_by_day(&[
            (ClickSat::Miss, DayOfWeek::Tue, 1),
            (ClickSat::Medium, DayOfWeek::Tue, 1),
            (ClickSat::Miss, DayOfWeek::Wed, 1),
            (ClickSat::Medium, DayOfWeek::Tue, 1),
            (ClickSat::Miss, DayOfWeek::Wed, 2),
            (ClickSat::Medium, DayOfWeek::Sun, 2),
            (ClickSat::Medium, DayOfWeek::Mon, 2),
            (ClickSat::Skip, DayOfWeek::Sun, 1),
            (ClickSat::Medium, DayOfWeek::Thu, 1),
            (ClickSat::Skip, DayOfWeek::Thu, 2),
            (ClickSat::Medium, DayOfWeek::Wed, 1),
            (ClickSat::Medium, DayOfWeek::Tue, 2),
            (ClickSat::Medium, DayOfWeek::Wed, 2),
            (ClickSat::Medium, DayOfWeek::Mon, 2),
            (ClickSat::Medium, DayOfWeek::Sat, 1),
            (ClickSat::Medium, DayOfWeek::Mon, 1),
            (ClickSat::Medium, DayOfWeek::Sat, 1),
            (ClickSat::Medium, DayOfWeek::Mon, 1),
        ]);

        // seasonality = (5*(1+w_end_day))/(2*(1+working_day))
        // relevant thu/fr days: 1
        // relevant other days: 7
        let value = seasonality(&history, 1);
        assert_approx_eq!(f32, value, 0.625);

        // relevant thu/fr days: 0
        // relevant other days: 5
        let value = seasonality(&history, 2);
        assert_approx_eq!(f32, value, 0.416_666_66);

        assert_approx_eq!(f32, seasonality(&[], 1), 0.0, ulps = 0);
    }

    impl DayOfWeek {
        fn create_test_day(day: u8) -> Self {
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

    fn query_results_by_words_and_session<'a>(
        iter: impl IntoIterator<Item = &'a (Vec<i32>, i32)>,
    ) -> Vec<SearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (query_words, session))| {
                let id = id as i32;
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                let query_counter = per_query_id as u8;

                SearchResult {
                    session_id: *session,
                    user_id: 1,
                    query_id: per_query_id,
                    day: DayOfWeek::create_test_day(query_counter),
                    query_words: query_words.clone(),
                    url: id,
                    domain: id,
                    relevance: ClickSat::Medium,
                    position: Rank::from_usize(1 + in_query_id as usize),
                    query_counter,
                }
            })
            .collect()
    }

    #[test]
    fn test_terms_variety() {
        let query = Query {
            id: 0,
            words: vec![3, 33, 12, 120, 33, 3],
        };
        assert_eq!(terms_variety(&query), 4);
    }

    fn query_results_by_rank_and_relevance<'a>(
        iter: impl IntoIterator<Item = &'a (Rank, ClickSat)>,
    ) -> Vec<SearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (rank, relevance))| {
                let id = id as i32;
                let per_query_id = id / 10;
                let query_counter = per_query_id as u8;

                SearchResult {
                    session_id: 1,
                    user_id: 1,
                    query_id: per_query_id,
                    day: DayOfWeek::create_test_day(query_counter),
                    query_words: vec![id],
                    url: id,
                    domain: id % 2,
                    relevance: *relevance,
                    position: *rank,
                    query_counter,
                }
            })
            .collect()
    }

    #[test]
    fn test_mean_reciprocal_rank() {
        let history = query_results_by_rank_and_relevance(&[
            (Rank::First, ClickSat::Skip),
            (Rank::Second, ClickSat::Miss),
            (Rank::Third, ClickSat::Low),
            (Rank::Fourth, ClickSat::Medium),
            (Rank::Fifth, ClickSat::High),
        ]);

        let mrr = mean_recip_rank(&history, None, None);
        assert_approx_eq!(f32, mrr, 0.427_722_25);

        let mrr = mean_recip_rank(&history, None, Some(FilterPred::new(UrlOrDom::Dom(0))));
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
            Some(FilterPred::new(UrlOrDom::Dom(0))),
        );
        assert_approx_eq!(f32, mrr, 0.241_5);
    }

    #[test]
    fn test_filter_predicate() {
        let filter = FilterPred::new(UrlOrDom::Dom(42));
        let mut result = SearchResult {
            session_id: 1,
            user_id: 2,
            query_id: 3,
            day: DayOfWeek::Sun,
            query_words: vec![1, 2],
            url: 11,
            domain: 22,
            relevance: ClickSat::High,
            position: Rank::First,
            query_counter: 1,
        };
        assert!(!filter.apply(&result));
        result.domain = 42;
        assert!(filter.apply(&result));

        let filter = filter.with_query(25);
        assert!(!filter.apply(&result));
        result.query_id = 25;
        assert!(filter.apply(&result));

        let filter = filter.with_session(SessionCond::Current(100));
        assert!(!filter.apply(&result));
        result.session_id = 100;
        assert!(filter.apply(&result));

        let filter = filter.with_session(SessionCond::Anterior(80));
        assert!(!filter.apply(&result));
        result.session_id = 80;
        assert!(!filter.apply(&result));
        result.session_id = 79;
        assert!(filter.apply(&result));

        let filter = filter.with_session(SessionCond::All);
        result.session_id = 3333;
        assert!(filter.apply(&result));

        let filter = FilterPred::new(UrlOrDom::Url(42));
        assert!(!filter.apply(&result));
        result.url = 42;
        assert!(filter.apply(&result));
    }

    #[test]
    fn the_right_cum_atoms_are_chosen() {
        use UrlOrDom::*;

        // No rust formatting for readability.
        // This is formatted as a table, which rustfmt would break.
        #[rustfmt::skip]
        let test_cases = vec![
            /* url.usual */
            (Url(1),    None,       SessionCond::All,           vec![ClickSat::Skip, ClickSat::Medium, ClickSat::High]),
            /* url.anterior */
            (Url(1),    None,       SessionCond::Anterior(10),  vec![]),
            /* url.session (not used) */
            (Url(1),    None,       SessionCond::Current(3),    vec![]),
            /* url.query */
            (Url(1),    Some(32),   SessionCond::All,           vec![]),
            /* url.query_anterior */
            (Url(1),    Some(32),   SessionCond::Anterior(10),  vec![]),
            /* url.query_session */
            (Url(1),    Some(32),   SessionCond::Current(3),    vec![]),
            /* domain.usual */
            (Dom(1),    None,       SessionCond::All,           vec![]),
            /* domain.anterior */
            (Dom(1),    None,       SessionCond::Anterior(10),  vec![]),
            /* domain.session (not used) */
            (Dom(1),    None,       SessionCond::Current(3),    vec![]),
            /* domain.query */
            (Dom(1),    Some(32),   SessionCond::All,           vec![]),
            /* domain.anterior */
            (Dom(1),    Some(32),   SessionCond::Anterior(10),  vec![]),
            /* domain.query_session (not used) */
            (Dom(1),    Some(32),   SessionCond::Current(3),    vec![]),
        ];

        for (url_or_dom, query_filter, session_cond, expected) in test_cases.into_iter() {
            let mut filter = FilterPred::new(url_or_dom).with_session(session_cond);
            if let Some(query) = query_filter {
                filter = filter.with_query(query);
            }

            let agg_atoms = filter.cum_atoms();
            if agg_atoms.len() != expected.len()
                || !expected.iter().all(|atom| agg_atoms.contains(atom))
            {
                panic!(
                    "for ({:?}, {:?}, {:?}) expected {:?} but got {:?}",
                    url_or_dom, query_filter, session_cond, expected, agg_atoms
                );
            }
        }
    }

    #[test]
    fn the_right_aggregate_atoms_are_chosen() {
        use UrlOrDom::*;

        let click_mrr = AtomFeat::MeanRecipRank(MrrOutcome::Click);
        let miss_mrr = AtomFeat::MeanRecipRank(MrrOutcome::Miss);
        let skip_mrr = AtomFeat::MeanRecipRank(MrrOutcome::Skip);
        let combine_mrr = AtomFeat::MeanRecipRankAll;
        let click2 = AtomFeat::CondProb(ClickSat::High);
        let missed = AtomFeat::CondProb(ClickSat::Miss);
        let skipped = AtomFeat::CondProb(ClickSat::Skip);
        let snipped_quality = AtomFeat::SnippetQuality;

        // No rust formatting for readability.
        // This is formatted as a table, which rustfmt would break.
        #[rustfmt::skip]
        let test_cases = vec![
            /* url.usual */
            (Url(1),    None,       SessionCond::All,           vec![click_mrr, click2, missed, snipped_quality]),
            /* url.anterior */
            (Url(1),    None,       SessionCond::Anterior(10),  vec![click2, missed, snipped_quality]),
            /* url.session (not used) */
            (Url(1),    None,       SessionCond::Current(3),    vec![]),
            /* url.query */
            (Url(1),    Some(32),   SessionCond::All,           vec![combine_mrr, click2, missed, snipped_quality]),
            /* url.query_anterior */
            (Url(1),    Some(32),   SessionCond::Anterior(10),  vec![combine_mrr, click_mrr, miss_mrr, skip_mrr, missed, snipped_quality]),
            /* url.query_session */
            (Url(1),    Some(32),   SessionCond::Current(3),    vec![miss_mrr]),
            /* domain.usual */
            (Dom(1),    None,       SessionCond::All,           vec![skipped, missed, click2, snipped_quality]),
            /* domain.anterior */
            (Dom(1),    None,       SessionCond::Anterior(10),  vec![click2, missed, snipped_quality]),
            /* domain.session (not used) */
            (Dom(1),    None,       SessionCond::Current(3),    vec![]),
            /* domain.query */
            (Dom(1),    Some(32),   SessionCond::All,           vec![missed, snipped_quality, miss_mrr]),
            /* domain.anterior */
            (Dom(1),    Some(32),   SessionCond::Anterior(10),  vec![snipped_quality]),
            /* domain.query_session (not used) */
            (Dom(1),    Some(32),   SessionCond::Current(3),    vec![]),
        ];

        for (url_or_dom, query_filter, session_cond, expected) in test_cases.into_iter() {
            let mut filter = FilterPred::new(url_or_dom).with_session(session_cond);
            if let Some(query) = query_filter {
                filter = filter.with_query(query);
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
    }

    #[test]
    fn test_is_clicked() {
        let mut result = SearchResult {
            session_id: 1,
            user_id: 1,
            query_id: 1,
            day: DayOfWeek::Fri,
            query_words: vec![1],
            url: 1,
            domain: 1,
            relevance: ClickSat::Skip,
            position: Rank::Eighth,
            query_counter: 1,
        };
        assert!(!result.is_clicked());

        result.relevance = ClickSat::Miss;
        assert!(!result.is_clicked());

        //FIXME clarify what Low means see comment at the top of the file
        result.relevance = ClickSat::Low;
        assert!(!result.is_clicked());

        result.relevance = ClickSat::Medium;
        assert!(result.is_clicked());

        result.relevance = ClickSat::High;
        assert!(result.is_clicked());
    }

    //TODO turn this into end to end tests where you compare the extracted feature arrays (once we do create them)
    //rustfmt makes all the assert_approx_eq much less readable in this specific case.
    #[rustfmt::skip]
    fn do_test_compute_features(history: &[SearchResult], query: &Query, search_results: &[CurrentSearchResult], features: &[[f32;50]]) {
        let user_features = UserFeatures::extract(history);
        let query_features = QueryFeatures::exact(history, query);
        let terms_variety = terms_variety(query);

        let click_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Click);
        let miss_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Miss);
        let skip_mrr = &AtomFeat::MeanRecipRank(MrrOutcome::Skip);
        let combine_mrr = &AtomFeat::MeanRecipRankAll;
        let click1 = &AtomFeat::CondProb(ClickSat::Medium);
        let click2 = &AtomFeat::CondProb(ClickSat::High);
        let missed = &AtomFeat::CondProb(ClickSat::Miss);
        let skipped = &AtomFeat::CondProb(ClickSat::Skip);
        let snippet_quality = &AtomFeat::SnippetQuality;

        for (search_result, feature_row) in izip!(search_results, features) {

            assert_approx_eq!(f32, feature_row[0], search_result.initial_rank as usize as f32, ulps = 0);

            let document_features = AggregateFeatures::extract(history, search_result);
            assert_approx_eq!(f32, feature_row[1], document_features.url[click_mrr]);
            assert_approx_eq!(f32, feature_row[2], document_features.url[click2]);
            assert_approx_eq!(f32, feature_row[3], document_features.url[missed]);
            assert_approx_eq!(f32, feature_row[4], document_features.url[snippet_quality]);
            assert_approx_eq!(f32, feature_row[5], document_features.url_ant[click2]);
            assert_approx_eq!(f32, feature_row[6], document_features.url_ant[missed]);
            assert_approx_eq!(f32, feature_row[7], document_features.url_ant[snippet_quality]);
            assert_approx_eq!(f32, feature_row[8], document_features.url_query[combine_mrr]);
            assert_approx_eq!(f32, feature_row[9], document_features.url_query[click2]);
            assert_approx_eq!(f32, feature_row[10], document_features.url_query[missed]);
            assert_approx_eq!(f32, feature_row[11], document_features.url_query[snippet_quality]);
            assert_approx_eq!(f32, feature_row[12], document_features.url_query_ant[combine_mrr]);
            assert_approx_eq!(f32, feature_row[13], document_features.url_query_ant[click_mrr]);
            assert_approx_eq!(f32, feature_row[14], document_features.url_query_ant[miss_mrr]);
            assert_approx_eq!(f32, feature_row[15], document_features.url_query_ant[skip_mrr]);
            assert_approx_eq!(f32, feature_row[16], document_features.url_query_ant[missed]);
            assert_approx_eq!(f32, feature_row[17], document_features.url_query_ant[snippet_quality]);
            assert_approx_eq!(f32, feature_row[18], document_features.url_query_curr[miss_mrr]);
            assert_approx_eq!(f32, feature_row[19], document_features.dom[skipped]);
            assert_approx_eq!(f32, feature_row[20], document_features.dom[missed]);
            assert_approx_eq!(f32, feature_row[21], document_features.dom[click2]);
            assert_approx_eq!(f32, feature_row[22], document_features.dom[snippet_quality]);
            assert_approx_eq!(f32, feature_row[23], document_features.dom_ant[click2]);
            assert_approx_eq!(f32, feature_row[24], document_features.dom_ant[missed]);
            assert_approx_eq!(f32, feature_row[25], document_features.dom_ant[snippet_quality]);
            assert_approx_eq!(f32, feature_row[26], document_features.dom_query[missed]);
            assert_approx_eq!(f32, feature_row[27], document_features.dom_query[snippet_quality]);
            assert_approx_eq!(f32, feature_row[28], document_features.dom_query[miss_mrr]);
            assert_approx_eq!(f32, feature_row[29], document_features.dom_query_ant[snippet_quality]);

            let QueryFeatures {
                click_entropy,
                num_terms,
                mean_query_counter,
                mean_occurs_per_session,
                num_occurs,
                click_mrr,
                mean_clicks,
                mean_non_click,
            } = query_features;
            assert_approx_eq!(f32, feature_row[30], click_entropy);
            assert_approx_eq!(f32, feature_row[31], num_terms as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[32], mean_query_counter);
            assert_approx_eq!(f32, feature_row[33], mean_occurs_per_session);
            assert_approx_eq!(f32, feature_row[34], num_occurs as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[35], click_mrr);
            assert_approx_eq!(f32, feature_row[36], mean_clicks);
            assert_approx_eq!(f32, feature_row[37], mean_non_click);

            let UserFeatures {
                click_entropy,
                click_counts,
                num_queries,
                mean_words_per_query,
                mean_unique_words_per_session,
            } = &user_features;
            assert_approx_eq!(f32, feature_row[38], click_entropy);
            assert_approx_eq!(f32, feature_row[39], click_counts.click12 as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[40], click_counts.click345 as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[41], click_counts.click6up as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[42], *num_queries as f32, ulps = 0);
            assert_approx_eq!(f32, feature_row[43], mean_words_per_query);
            assert_approx_eq!(f32, feature_row[44], mean_unique_words_per_session);

            let cum_features = CumFeatures::extract(history, search_result);
            //FIXME that fails (as expected)
            assert_approx_eq!(f32, feature_row[45], cum_features.url[skipped]);
            assert_approx_eq!(f32, feature_row[46], cum_features.url[click1]);
            assert_approx_eq!(f32, feature_row[47], cum_features.url[click2]);

            assert_approx_eq!(f32, feature_row[48], terms_variety as f32, ulps = 0);

            let seasonality = seasonality(history, search_result.domain);
            assert_approx_eq!(f32, feature_row[49], seasonality);
        }
    }

    #[test]
    fn test_full_training_1() {
        //auto generated
        #[rustfmt::skip]
        let history = &[
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 26142648, domain: 2597528, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 44200215, domain: 3852697, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 40218620, domain: 3602893, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 21854374, domain: 2247911, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 6152223, domain: 787424, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 46396840, domain: 3965502, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 65705884, domain: 4978404, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 4607041, domain: 608358, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 60306140, domain: 4679885, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 12283852, day: DayOfWeek::Fri, query_words: vec![3468976,4614115], url: 1991065, domain: 295576, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 43220173, domain: 3802280, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 68391867, domain: 5124172, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 48241082, domain: 4077775, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 28461283, domain: 2809381, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 36214392, domain: 3398386, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 26215090, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 55157032, domain: 4429726, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 35921251, domain: 3380635, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 37498049, domain: 3463275, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746324, user_id: 460950, query_id: 7297472, day: DayOfWeek::Fri, query_words: vec![2758230], url: 70173304, domain: 5167485, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
        ];

        //auto generated
        #[rustfmt::skip]
        let current_search_results = &[
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 41131641, domain: 3661944, initial_rank: Rank::from_usize(1), query_counter: 0, },
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 43630521, domain: 3823198, initial_rank: Rank::from_usize(2), query_counter: 0, },
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 28819788, domain: 2832997, initial_rank: Rank::from_usize(3), query_counter: 0, },
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 28630417, domain: 2819308, initial_rank: Rank::from_usize(4), query_counter: 0, },
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 49489872, domain: 4155543, initial_rank: Rank::from_usize(5), query_counter: 0, },
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 1819187, domain: 269174, initial_rank: Rank::from_usize(6), query_counter: 0, },
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 27680026, domain: 2696111, initial_rank: Rank::from_usize(7), query_counter: 0, },
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 1317174, domain: 207936, initial_rank: Rank::from_usize(8), query_counter: 0, },
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 28324834, domain: 2790971, initial_rank: Rank::from_usize(9), query_counter: 0, },
            CurrentSearchResult { session_id: 2746325, user_id: 460950, query_id: 20331734, day: DayOfWeek::Fri, query_words: vec![4631619,2289501], url: 54208271, domain: 4389621, initial_rank: Rank::from_usize(10), query_counter: 0, },
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

        let query = Query {
            id: current_search_results[0].query_id,
            words: current_search_results[0].query_words.clone(),
        };
        do_test_compute_features(history, &query, current_search_results, features);
    }

    #[test]
    fn test_full_training_2() {
        //auto generated
        #[rustfmt::skip]
        let history = &[
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 53147190, domain: 4333384, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 35139928, domain: 3310380, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 41556099, domain: 3694829, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 41551846, domain: 3694697, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 43953712, domain: 3841362, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 17411412, domain: 1830842, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 41554891, domain: 3694807, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 14667292, domain: 1509151, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 4613488, domain: 609534, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746384, user_id: 460957, query_id: 9825574, day: DayOfWeek::Tue, query_words: vec![3047095,4696813,4117921,3539220,2783606], url: 53150986, domain: 4333466, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 53663868, domain: 4361547, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 49901944, domain: 4189192, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 22545102, domain: 2258385, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 41465426, domain: 3689897, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 35864630, domain: 3376770, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 39904753, domain: 3594796, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 58600901, domain: 4596609, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 55384497, domain: 4437568, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 68342291, domain: 5121329, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 10852271, day: DayOfWeek::Wed, query_words: vec![3199658,3442492,2766737,2542435], url: 47101006, domain: 4000131, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 3320136, domain: 450234, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 49799616, domain: 4178302, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 20703572, domain: 2108878, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 51167490, domain: 4239935, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 12548156, domain: 1261523, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 8420371, domain: 1005265, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 35411186, domain: 3337284, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 58594730, domain: 4596075, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 38319703, domain: 3505936, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746385, user_id: 460957, query_id: 18586669, day: DayOfWeek::Wed, query_words: vec![4347441,3442492,2766737,2542435,284354], url: 48940975, domain: 4127329, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 53629459, domain: 4361365, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 58311620, domain: 4583436, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 65215277, domain: 4958818, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 58679732, domain: 4599667, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 53456347, domain: 4357388, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 68698823, domain: 5142433, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 8263067, domain: 985006, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 38268355, domain: 3503180, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 17831079, domain: 1882981, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740435, day: DayOfWeek::Wed, query_words: vec![3006121,2971595], url: 63857792, domain: 4863475, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 33632736, domain: 3248439, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 22668397, domain: 2278222, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 52271655, domain: 4295885, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 15469473, domain: 1598176, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 3584882, domain: 482553, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 35869091, domain: 3376915, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 35179383, domain: 3312539, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 52624099, domain: 4297303, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 71027328, domain: 5278963, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 10852326, day: DayOfWeek::Wed, query_words: vec![3199658,4553172], url: 3718435, domain: 491815, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 39385381, domain: 3562233, relevance: ClickSat::Medium, position: Rank::from_usize(1), query_counter: 2, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 23179149, domain: 2330832, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 2, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 58308533, domain: 4583436, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 2, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 60819582, domain: 4705737, relevance: ClickSat::High, position: Rank::from_usize(4), query_counter: 2, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 62746728, domain: 4819238, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 2, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 68292977, domain: 5121069, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 2, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 55789105, domain: 4448304, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 2, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 53629157, domain: 4361365, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 2, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 39424689, domain: 3562849, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 2, },
            SearchResult { session_id: 2746386, user_id: 460957, query_id: 8740434, day: DayOfWeek::Wed, query_words: vec![3006121,2379505], url: 22162115, domain: 2254454, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 2, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 37860644, domain: 3479189, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 26445606, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 7663723, domain: 943850, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 1579586, domain: 228599, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 39699867, domain: 3578476, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 16285104, domain: 1697410, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 15475166, domain: 1599137, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 7243881, domain: 893309, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 32168416, domain: 3142702, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 689945, day: DayOfWeek::Thu, query_words: vec![547388], url: 1578848, domain: 228424, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 37778519, domain: 3473725, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 37778529, domain: 3473725, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 49420327, domain: 4150039, relevance: ClickSat::High, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 39362290, domain: 3562233, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 53635244, domain: 4361365, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 17744438, domain: 1878457, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 38264285, domain: 3503180, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 60750918, domain: 4701228, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 57365833, domain: 4533325, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746387, user_id: 460957, query_id: 6036733, day: DayOfWeek::Thu, query_words: vec![2513115,2267668,3309815,2903008], url: 58309258, domain: 4583436, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 46149482, domain: 3956637, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 41301331, domain: 3677581, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 15857322, domain: 1642802, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 8145477, domain: 968515, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 27202128, domain: 2637320, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 27219158, domain: 2637320, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 50014375, domain: 4192304, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 21019235, domain: 2152185, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 9234110, domain: 1073438, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 58655249, domain: 4598966, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 68792294, domain: 5147706, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 63230596, domain: 4842497, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 63230259, domain: 4842497, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 70719509, domain: 5238396, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 39175442, domain: 3556153, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 39176001, domain: 3556153, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 62705968, domain: 4816695, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 29456421, domain: 2863832, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 53348354, domain: 4348568, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15754654, day: DayOfWeek::Fri, query_words: vec![4022204,2226531], url: 47491280, domain: 4025997, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 50013429, domain: 4192244, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 35140109, domain: 3310380, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 54690595, domain: 4412120, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 41551463, domain: 3694697, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 35008132, domain: 3297128, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 31770039, domain: 3106054, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 56890641, domain: 4503208, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 46910925, domain: 3993171, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 32975326, domain: 3223851, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15762378, day: DayOfWeek::Fri, query_words: vec![4022204,2226531,4696663], url: 25828504, domain: 2586750, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 2, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 46149482, domain: 3956637, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 41301331, domain: 3677581, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 15857322, domain: 1642802, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 8145477, domain: 968515, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 27202128, domain: 2637320, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 27219158, domain: 2637320, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 50014375, domain: 4192304, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 21019235, domain: 2152185, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 9234110, domain: 1073438, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 10897212, day: DayOfWeek::Fri, query_words: vec![3209440,2780329,3442492,4175110], url: 58655249, domain: 4598966, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 3, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 24945706, domain: 2497978, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 4, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 24946072, domain: 2497978, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 4, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 66148501, domain: 5003733, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 4, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 37735784, domain: 3471243, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 4, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 12533465, domain: 1259258, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 4, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 4971996, domain: 648015, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 4, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 13702454, domain: 1406295, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 4, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 61662925, domain: 4741007, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 4, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 66188557, domain: 5004729, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 4, },
            SearchResult { session_id: 2746388, user_id: 460957, query_id: 15803639, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4826385], url: 25002604, domain: 2504335, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 4, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 13954927, domain: 1424288, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 41736799, domain: 3708167, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 61249759, domain: 4723745, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 70963808, domain: 5269793, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 26963737, domain: 2617682, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 55058001, domain: 4423590, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 43010696, domain: 3792414, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 3727676, domain: 492823, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 67373261, domain: 5070345, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5037935, day: DayOfWeek::Fri, query_words: vec![2318509,3085676], url: 34478692, domain: 3279130, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 67224468, domain: 5058813, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 67965841, domain: 5099408, relevance: ClickSat::High, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 58309583, domain: 4583436, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 69994859, domain: 5158848, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 60758683, domain: 4701228, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 47949400, domain: 4049883, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 44417566, domain: 3861332, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 58987072, domain: 4615084, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 6296543, domain: 787424, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 5038604, day: DayOfWeek::Fri, query_words: vec![2318792,3085676,4117921,2646723,2452511,4385171], url: 26041333, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 11926173, domain: 1197084, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 26071404, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 70236237, domain: 5169937, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 47047062, domain: 3997492, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 47412349, domain: 4023234, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 68603751, domain: 5138848, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 52100903, domain: 4288663, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 53768788, domain: 4364544, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 6296981, domain: 787424, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 6764704, day: DayOfWeek::Fri, query_words: vec![2646661], url: 19449102, domain: 2018255, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 2, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 35140109, domain: 3310380, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 3, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 24946954, domain: 2497978, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 3, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 37736024, domain: 3471243, relevance: ClickSat::High, position: Rank::from_usize(3), query_counter: 3, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 17286928, domain: 1810971, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 3, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 66142938, domain: 5003733, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 3, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 15616993, domain: 1609843, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 3, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 41551741, domain: 3694697, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 3, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 61196252, domain: 4722065, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 3, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 63764585, domain: 4855871, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 3, },
            SearchResult { session_id: 2746389, user_id: 460957, query_id: 15803568, day: DayOfWeek::Fri, query_words: vec![4022204,4782226,4696627], url: 35140683, domain: 3310382, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 3, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 11926173, domain: 1197084, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 26071404, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 70236237, domain: 5169937, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 47047062, domain: 3997492, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 47412349, domain: 4023234, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 68603751, domain: 5138848, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 52100903, domain: 4288663, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 53768788, domain: 4364544, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 6296981, domain: 787424, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746390, user_id: 460957, query_id: 6764704, day: DayOfWeek::Sun, query_words: vec![2646661], url: 19449102, domain: 2018255, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 48946911, domain: 4127959, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 12848897, domain: 1302254, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 55791474, domain: 4448356, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 3257558, domain: 442452, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 55394186, domain: 4438002, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 7253853, domain: 894252, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 6486578, domain: 802423, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 26074227, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 16582011, domain: 1718183, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746391, user_id: 460957, query_id: 6953271, day: DayOfWeek::Wed, query_words: vec![2675783,4617765], url: 50260384, domain: 4206145, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 53087314, domain: 4330106, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 26128734, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 6027697, domain: 787424, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 13643279, domain: 1399059, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 59239047, domain: 4627364, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 17358783, domain: 1822941, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 28466565, domain: 2809381, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 16914467, domain: 1765561, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 24121106, domain: 2406385, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746392, user_id: 460957, query_id: 11342686, day: DayOfWeek::Thu, query_words: vec![3309299], url: 28065878, domain: 2754488, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 59584008, domain: 4642380, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 24503960, domain: 2445674, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 24501260, domain: 2445287, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 24501305, domain: 2445287, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 24500104, domain: 2445258, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 6437479, domain: 796433, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 24493224, domain: 2444820, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 59580110, domain: 4642280, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 59569247, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746393, user_id: 460957, query_id: 8004415, day: DayOfWeek::Sat, query_words: vec![2886907,1607252,2931909,3442492,3962093,412363,1401653], url: 24494769, domain: 2444971, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 475038, domain: 79639, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 59580013, domain: 4642280, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 59580073, domain: 4642280, relevance: ClickSat::High, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 59583619, domain: 4642380, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 24501226, domain: 2445287, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 11716332, domain: 1165300, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 24504280, domain: 2445674, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 24492136, domain: 2444754, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 59569301, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746394, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sat, query_words: vec![1585181,412363], url: 24498739, domain: 2445239, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 475038, domain: 79639, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 59580013, domain: 4642280, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 59580073, domain: 4642280, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 59583619, domain: 4642380, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 24501226, domain: 2445287, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 11716332, domain: 1165300, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 24504280, domain: 2445674, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 24492136, domain: 2444754, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 59569301, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 24498739, domain: 2445239, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 475038, domain: 79639, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 59580013, domain: 4642280, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 59580073, domain: 4642280, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 59583619, domain: 4642380, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 24501226, domain: 2445287, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 11716332, domain: 1165300, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 24504280, domain: 2445674, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 24492136, domain: 2444754, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 59569301, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 3171117, day: DayOfWeek::Sun, query_words: vec![1585181,412363], url: 24498739, domain: 2445239, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 69827886, domain: 5157997, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 2, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 19373740, domain: 2017204, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 2, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 71176046, domain: 5301431, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 2, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 41994651, domain: 3718651, relevance: ClickSat::Skip, position: Rank::from_usize(4), query_counter: 2, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 3173415, domain: 438941, relevance: ClickSat::High, position: Rank::from_usize(5), query_counter: 2, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 43649530, domain: 3825045, relevance: ClickSat::Skip, position: Rank::from_usize(6), query_counter: 2, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 60109238, domain: 4673119, relevance: ClickSat::High, position: Rank::from_usize(7), query_counter: 2, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 18993172, domain: 1993793, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 2, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 28308269, domain: 2790282, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 2, },
            SearchResult { session_id: 2746395, user_id: 460957, query_id: 12000844, day: DayOfWeek::Sun, query_words: vec![3423840,4219157,2383044], url: 56290971, domain: 4468792, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 2, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 31734981, domain: 3103194, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 2704116, domain: 382366, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 65790152, domain: 4983465, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 31731943, domain: 3102982, relevance: ClickSat::High, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 31732996, domain: 3103147, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 65792028, domain: 4983560, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 43635792, domain: 3823536, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 12058291, domain: 1203722, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 4927129, domain: 647384, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746396, user_id: 460957, query_id: 12002261, day: DayOfWeek::Sun, query_words: vec![3423840,4462772,2383044], url: 19983204, domain: 2038964, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 24504280, domain: 2445674, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 59579711, domain: 4642280, relevance: ClickSat::Medium, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 59580013, domain: 4642280, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 44780109, domain: 3884841, relevance: ClickSat::Skip, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 59581661, domain: 4642352, relevance: ClickSat::Skip, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 475038, domain: 79639, relevance: ClickSat::Skip, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 24498739, domain: 2445239, relevance: ClickSat::Skip, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 59581381, domain: 4642304, relevance: ClickSat::High, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 24492038, domain: 2444754, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746397, user_id: 460957, query_id: 3171118, day: DayOfWeek::Sun, query_words: vec![1585181,924909,412363], url: 24501226, domain: 2445287, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 59569799, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 25993337, domain: 2596633, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 69267864, domain: 5149883, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 24533646, domain: 2447734, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 31082234, domain: 3038166, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 29011305, domain: 2844548, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 24492691, domain: 2444769, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 53002186, domain: 4323004, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 20876888, domain: 2133879, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 3219904, day: DayOfWeek::Mon, query_words: vec![1606510], url: 49860164, domain: 4184441, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 20126923, domain: 2056301, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 12699201, domain: 1284324, relevance: ClickSat::High, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 14268148, domain: 1468684, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 4190926, domain: 550386, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 2306887, domain: 342665, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 2306886, domain: 342665, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 28534999, domain: 2811267, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 18625148, domain: 1958100, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 18625145, domain: 1958100, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 15226059, day: DayOfWeek::Mon, query_words: vec![3930428,2767265,4221108,4117921,3466583], url: 34857095, domain: 3279950, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 48714333, domain: 4113544, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 7059666, domain: 867291, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 46774237, domain: 3987015, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 48713490, domain: 4113544, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 5497729, domain: 709167, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 34548399, domain: 3279130, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 42942542, domain: 3788194, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 12794211, domain: 1297117, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 68362077, domain: 5123163, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1311973, day: DayOfWeek::Mon, query_words: vec![778177,1778288], url: 17670553, domain: 1871818, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 2, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 24498060, domain: 2445222, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 24452682, domain: 2441201, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 11715886, domain: 1165300, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 24486376, domain: 2444509, relevance: ClickSat::Skip, position: Rank::from_usize(4), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 24504095, domain: 2445674, relevance: ClickSat::High, position: Rank::from_usize(5), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 24488273, domain: 2444626, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 25993756, domain: 2596633, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 24484495, domain: 2444483, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 59575416, domain: 4642280, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309882, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1606572], url: 24500251, domain: 2445258, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 3, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 43905844, domain: 3838097, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 4, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 43905847, domain: 3838097, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 4, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 25993756, domain: 2596633, relevance: ClickSat::High, position: Rank::from_usize(3), query_counter: 4, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 11468954, domain: 1143418, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 4, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 42681664, domain: 3765450, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 4, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 24448467, domain: 2441087, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 4, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 5725247, domain: 745552, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 4, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 8251942, domain: 983193, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 4, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 40809252, domain: 3641926, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 4, },
            SearchResult { session_id: 2746398, user_id: 460957, query_id: 1309881, day: DayOfWeek::Mon, query_words: vec![777467,1773027,3442492,1605549], url: 11477932, domain: 1143850, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 4, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 11580107, domain: 1147271, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 28541325, domain: 2811680, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 11484813, domain: 1143888, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 46723841, domain: 3986153, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 58530008, domain: 4592974, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 37245456, domain: 3444061, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 7692001, domain: 943850, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 26450035, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 61493555, domain: 4729342, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 1027807, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024], url: 11590334, domain: 1147673, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 59569559, domain: 4642238, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 59537662, domain: 4640614, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 24490217, domain: 2444643, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 24499933, domain: 2445258, relevance: ClickSat::Skip, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 43027515, domain: 3792852, relevance: ClickSat::Skip, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 24489755, domain: 2444626, relevance: ClickSat::Skip, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 45710517, domain: 3933182, relevance: ClickSat::Skip, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 59584101, domain: 4642380, relevance: ClickSat::Skip, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 5178707, domain: 668826, relevance: ClickSat::Skip, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746399, user_id: 460957, query_id: 2868337, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1607255], url: 59571219, domain: 4642273, relevance: ClickSat::High, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 37138421, domain: 3438838, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 57416404, domain: 4536418, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 45141304, domain: 3906237, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 62415804, domain: 4800893, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 62415864, domain: 4800893, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 58729203, domain: 4601864, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 67820903, domain: 5088851, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 53724644, domain: 4363170, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 32677380, domain: 3180707, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 6502577, day: DayOfWeek::Mon, query_words: vec![2601736,4750626,2926659,3362859,3649428,226242], url: 42890265, domain: 3783410, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 4305621, domain: 571229, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 5697749, domain: 742074, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 11482372, domain: 1143888, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 26450035, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 11590334, domain: 1147673, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 3225689, domain: 441339, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 3239168, domain: 441339, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 11579429, domain: 1147271, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 31087925, domain: 3038166, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 1027808, day: DayOfWeek::Mon, query_words: vec![670705,318563,671114,1495024,3442492,3966942], url: 20357359, domain: 2093274, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 20514173, domain: 2105616, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 35271558, domain: 3321740, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 61933977, domain: 4765984, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 67980432, domain: 5100287, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 20696227, domain: 2108209, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 49208605, domain: 4139272, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 31466951, domain: 3084870, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 56198234, domain: 4466995, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 58731528, domain: 4601864, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016804, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042], url: 18184774, domain: 1910044, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 2, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 61933977, domain: 4765984, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 18184774, domain: 1910044, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 31466951, domain: 3084870, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 22847626, domain: 2301541, relevance: ClickSat::Skip, position: Rank::from_usize(4), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 18239760, domain: 1923028, relevance: ClickSat::High, position: Rank::from_usize(5), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 57418635, domain: 4536474, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 15721549, domain: 1622852, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 56251095, domain: 4468463, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 58731529, domain: 4601864, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016814, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3600300], url: 67127036, domain: 5054761, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 3, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 15392058, domain: 1585727, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 35271558, domain: 3321740, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 67127036, domain: 5054761, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 60632747, domain: 4693249, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 22847484, domain: 2301541, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 22847626, domain: 2301541, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 49208605, domain: 4139272, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 61933977, domain: 4765984, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 58731529, domain: 4601864, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8016815, day: DayOfWeek::Mon, query_words: vec![2889662,3825500,2452511,286862,2652042,3678520,2452511,4032349], url: 67193606, domain: 5056248, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 4, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 58731527, domain: 4601864, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 58731529, domain: 4601864, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 66987070, domain: 5047561, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 30340447, domain: 2939605, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 39521468, domain: 3569000, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 15721549, domain: 1622852, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 56704719, domain: 4491825, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 44791233, domain: 3885561, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 28779437, domain: 2829538, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 8879142, day: DayOfWeek::Mon, query_words: vec![3018658,2588163,2889749,3825500,2452511,286862,2652042], url: 8890735, domain: 1040501, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 5, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 29865699, domain: 2898669, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 6, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 22847626, domain: 2301541, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 6, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 22847730, domain: 2301541, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 6, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 60304473, domain: 4679698, relevance: ClickSat::High, position: Rank::from_usize(4), query_counter: 6, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 58999720, domain: 4616072, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 6, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 50777132, domain: 4224168, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 6, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 68014849, domain: 5101478, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 6, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 56257366, domain: 4468463, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 6, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 20074779, domain: 2047873, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 6, },
            SearchResult { session_id: 2746400, user_id: 460957, query_id: 9336011, day: DayOfWeek::Mon, query_words: vec![3018658,4028621,2889749,3825500,2452511,286862,2652042], url: 18933364, domain: 1991401, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 6, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 22847626, domain: 2301541, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 56247982, domain: 4468463, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 35271558, domain: 3321740, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 22566860, domain: 2261650, relevance: ClickSat::Skip, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 67127036, domain: 5054761, relevance: ClickSat::Skip, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 11390530, domain: 1133377, relevance: ClickSat::Skip, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 61933977, domain: 4765984, relevance: ClickSat::Skip, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 58731529, domain: 4601864, relevance: ClickSat::High, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 58724397, domain: 4601863, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746401, user_id: 460957, query_id: 9302403, day: DayOfWeek::Mon, query_words: vec![3018658,3939272,4032300,2889765,3825500,2452511,286862,2652042], url: 60304473, domain: 4679698, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 23145467, domain: 2327307, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 42046072, domain: 3723151, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 31681868, domain: 3098306, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 42590444, domain: 3761190, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 42180151, domain: 3731837, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 27624204, domain: 2689571, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 25936771, domain: 2594853, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 55632684, domain: 4440802, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 4210161, domain: 552986, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637933, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913], url: 31234439, domain: 3054178, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 16683436, domain: 1735488, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 16683341, domain: 1735488, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 16046361, domain: 1674994, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 65636447, domain: 4975723, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 21326414, domain: 2187056, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 24719534, domain: 2468331, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 2228724, domain: 330060, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 61459146, domain: 4728402, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 4600186, domain: 607917, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 15729440, day: DayOfWeek::Wed, query_words: vec![4016942,4117921,3493749,3442492,1605532], url: 35787683, domain: 3366630, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 10849029, domain: 1094456, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 4946772, domain: 648015, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 2062142, domain: 305344, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 62398773, domain: 4799153, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 49865575, domain: 4185346, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 10849146, domain: 1094470, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 53345149, domain: 4348319, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 58323192, domain: 4584597, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 59581488, domain: 4642322, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 1637934, day: DayOfWeek::Wed, query_words: vec![908162,515157,783913,1607273], url: 10760794, domain: 1093709, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 2, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 24504996, domain: 2445721, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 59576515, domain: 4642280, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 24499911, domain: 2445258, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 59569215, domain: 4642238, relevance: ClickSat::High, position: Rank::from_usize(4), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 24504985, domain: 2445719, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 26796229, domain: 2617582, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 24489585, domain: 2444626, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 24493871, domain: 2444850, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 24490411, domain: 2444645, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 9437069, day: DayOfWeek::Wed, query_words: vec![3018658,4267174,2933841,3649428,1607252,3442492,1606572], url: 59567924, domain: 4642195, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 3, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 55632684, domain: 4440802, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 4, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 9556512, domain: 1081045, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 4, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 63736570, domain: 4854362, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 4, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 29741058, domain: 2882269, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 4, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 19650036, domain: 2025957, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 4, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 3970700, domain: 519479, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 4, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 23145467, domain: 2327307, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 4, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 830374, domain: 133398, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 4, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 38418375, domain: 3511792, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 4, },
            SearchResult { session_id: 2746402, user_id: 460957, query_id: 14935980, day: DayOfWeek::Wed, query_words: vec![3899348,2993522,1607273,3870283,908162,515157,783913], url: 52182761, domain: 4291703, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 4, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24504989, domain: 2445721, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 58539851, domain: 4593027, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24488999, domain: 2444626, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 59569215, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24485633, domain: 2444509, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24500230, domain: 2445258, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 59567924, domain: 4642195, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 31094180, domain: 3038166, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24490411, domain: 2444645, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24496548, domain: 2445119, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24504989, domain: 2445721, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 58539851, domain: 4593027, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24488999, domain: 2444626, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 59569215, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24485633, domain: 2444509, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24500230, domain: 2445258, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 59567924, domain: 4642195, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 31094180, domain: 3038166, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24490411, domain: 2444645, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24496548, domain: 2445119, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24504989, domain: 2445721, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 58539851, domain: 4593027, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24488999, domain: 2444626, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 59569215, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24485633, domain: 2444509, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24500230, domain: 2445258, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 59567924, domain: 4642195, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 31094180, domain: 3038166, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24490411, domain: 2444645, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 18053624, day: DayOfWeek::Tue, query_words: vec![4266975,1607252,3442492,1606572], url: 24496548, domain: 2445119, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 2, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 63736570, domain: 4854362, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 3, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 55632684, domain: 4440802, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 3, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 9556512, domain: 1081045, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 3, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 29741058, domain: 2882269, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 3, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 4762379, domain: 632691, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 3, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 52177170, domain: 4291616, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 3, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 64577060, domain: 4903225, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 3, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 42590444, domain: 3761190, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 3, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 43811150, domain: 3832201, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 3, },
            SearchResult { session_id: 2746403, user_id: 460957, query_id: 14935979, day: DayOfWeek::Tue, query_words: vec![3899348,2993522,1607273,2535356,908162,515157,783913], url: 10083791, domain: 1086563, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 3, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 24504989, domain: 2445721, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 59574512, domain: 4642280, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 59569215, domain: 4642238, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 30560958, domain: 2976061, relevance: ClickSat::High, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 58543374, domain: 4593444, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 24493871, domain: 2444850, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 24489983, domain: 2444626, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 10230057, domain: 1088025, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 68799121, domain: 5147985, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9438683, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,3571137,1607252,3442492,1606572], url: 59583278, domain: 4642380, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 24504989, domain: 2445721, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 24499911, domain: 2445258, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 59567924, domain: 4642195, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 58539851, domain: 4593027, relevance: ClickSat::High, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 24488999, domain: 2444626, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 59569215, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 24496548, domain: 2445119, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 46762822, domain: 3986725, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 24493871, domain: 2444850, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746404, user_id: 460957, query_id: 9435100, day: DayOfWeek::Tue, query_words: vec![3018658,4267174,1607252,3442492,1606572], url: 59579199, domain: 4642280, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 24504989, domain: 2445721, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 24505037, domain: 2445721, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 44667521, domain: 3877102, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 30560958, domain: 2976061, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 59569215, domain: 4642238, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 59584008, domain: 4642380, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 59576515, domain: 4642280, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 24533561, domain: 2447734, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 31092679, domain: 3038166, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746405, user_id: 460957, query_id: 15219417, day: DayOfWeek::Tue, query_words: vec![3930428,2767265,3726908,3571176,1607252,2452511,1606572], url: 5246852, domain: 673931, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
        ];

        //auto generated
        #[rustfmt::skip]
        let current_search_results = &[
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 33149547, domain: 3247429, initial_rank: Rank::from_usize(1), query_counter: 0, },
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 5583756, domain: 723219, initial_rank: Rank::from_usize(2), query_counter: 0, },
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 5583757, domain: 723219, initial_rank: Rank::from_usize(3), query_counter: 0, },
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 63784236, domain: 4857582, initial_rank: Rank::from_usize(4), query_counter: 0, },
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 42712627, domain: 3767948, initial_rank: Rank::from_usize(5), query_counter: 0, },
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 51421459, domain: 4253487, initial_rank: Rank::from_usize(6), query_counter: 0, },
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 64747283, domain: 4918015, initial_rank: Rank::from_usize(7), query_counter: 0, },
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 527421, domain: 86560, initial_rank: Rank::from_usize(8), query_counter: 0, },
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 5581590, domain: 722853, initial_rank: Rank::from_usize(9), query_counter: 0, },
            CurrentSearchResult { session_id: 2746406, user_id: 460957, query_id: 18839892, day: DayOfWeek::Thu, query_words: vec![4394569,2452511,4334072,4566900,2926659,4643462,2499462], url: 42709855, domain: 3767675, initial_rank: Rank::from_usize(10), query_counter: 0, },
        ];

        //auto generated
        #[rustfmt::skip]
        let features = &[
            [1.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0],
            [2.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0],
            [3.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0],
            [4.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0],
            [5.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0],
            [6.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0],
            [7.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0],
            [8.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0],
            [9.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0],
            [10.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.283, 0.283, 0.283, 0.283, 0.0, 0.0, 0.283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.283, 0.0, 0.0, 2.596993155107387, 14.0, 13.0, 4.0, 54.0, 4.055555555555555, 7.2727272727272725, 0.0, 0.0, 0.0, 7.0, 0.0]
        ];

        let query = Query {
            id: current_search_results[0].query_id,
            words: current_search_results[0].query_words.clone(),
        };
        do_test_compute_features(history, &query, current_search_results, features);
    }

    #[test]
    fn test_full_training_3() {
        //auto generated
        #[rustfmt::skip]
        let history = &[
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 22720229, domain: 2287656, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 15789592, domain: 1633601, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 64298839, domain: 4887956, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 4953470, domain: 648015, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 1750908, domain: 258559, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 3166996, domain: 438460, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 1091643, domain: 164268, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 5873160, domain: 769945, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 852123, domain: 134468, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746329, user_id: 460953, query_id: 4918994, day: DayOfWeek::Wed, query_words: vec![2286851,3442492,3993360,4337840], url: 28535061, domain: 2811267, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 57041595, domain: 4513957, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 52633903, domain: 4297482, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 38606550, domain: 3528795, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 23339933, domain: 2346874, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 22673827, domain: 2279594, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 27837762, domain: 2720526, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 11845280, domain: 1185380, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 8439393, domain: 1006919, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 2230776, domain: 330257, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746330, user_id: 460953, query_id: 10384773, day: DayOfWeek::Wed, query_words: vec![3120268,2320171], url: 34678029, domain: 3279130, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 69827886, domain: 5157997, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 70943655, domain: 5267270, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 32129007, domain: 3140281, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 12336787, domain: 1233129, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 27132145, domain: 2631599, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 169532, domain: 37590, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 70512516, domain: 5211321, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 19556954, domain: 2021907, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 29103972, domain: 2854312, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 17367682, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 68962528, domain: 5149883, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 48339396, domain: 4084893, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 48338613, domain: 4084893, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 6988247, domain: 862345, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 39318369, domain: 3562125, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 45125817, domain: 3905377, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 15423198, domain: 1590655, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 68286584, domain: 5121069, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 15382691, domain: 1585249, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 52730298, domain: 4304719, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746331, user_id: 460953, query_id: 20622315, day: DayOfWeek::Wed, query_words: vec![4693850,3153990,3273880,4710744,3611299,3842151], url: 66312568, domain: 5009298, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 69827886, domain: 5157997, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 32129007, domain: 3140281, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 12336787, domain: 1233129, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 3257068, domain: 442410, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 35353234, domain: 3330660, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 35029384, domain: 3299856, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 60141213, domain: 4673119, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 28310481, domain: 2790282, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 169532, domain: 37590, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746332, user_id: 460953, query_id: 17370106, day: DayOfWeek::Wed, query_words: vec![4219157,3424615,2870281,3514644,2369241], url: 68962528, domain: 5149883, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 69857015, domain: 5157997, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 70943655, domain: 5267270, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 32129007, domain: 3140281, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 12336787, domain: 1233129, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 3257068, domain: 442410, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 35029384, domain: 3299856, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 60141213, domain: 4673119, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 28310481, domain: 2790282, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 169532, domain: 37590, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746333, user_id: 460953, query_id: 17370107, day: DayOfWeek::Thu, query_words: vec![4219157,3424615,2870281,3514644,2383044], url: 68962528, domain: 5149883, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 9728531, domain: 1083502, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 18685380, domain: 1964815, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 22286061, domain: 2254454, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 1326481, domain: 208839, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 13922954, domain: 1420922, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 50025724, domain: 4193303, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 59575505, domain: 4642280, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 3287258, domain: 444834, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 22495719, domain: 2254602, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746334, user_id: 460953, query_id: 10614893, day: DayOfWeek::Sat, query_words: vec![3162900,2320176], url: 2231421, domain: 330257, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 44087756, domain: 3844340, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 58531451, domain: 4592974, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 65600042, domain: 4975723, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 26468733, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 30015923, domain: 2917260, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 11486514, domain: 1143888, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 65530609, domain: 4975721, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 7820914, domain: 943850, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 20720355, domain: 2111503, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 2868275, day: DayOfWeek::Sun, query_words: vec![1468238,924909,1777674,1603687], url: 55954880, domain: 4457118, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 21218305, domain: 2176674, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 42696217, domain: 3766183, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 14611964, domain: 1500014, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 23285128, domain: 2342128, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 58536938, domain: 4592974, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 1365591, domain: 210625, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 23390674, domain: 2351344, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 11487985, domain: 1143888, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 28282311, domain: 2786993, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746335, user_id: 460953, query_id: 9841989, day: DayOfWeek::Sun, query_words: vec![3048434,1876555], url: 31872478, domain: 3114734, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 44087756, domain: 3844340, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 58531451, domain: 4592974, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 65600042, domain: 4975723, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 26468733, domain: 2597528, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 30015923, domain: 2917260, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 11486514, domain: 1143888, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 65530609, domain: 4975721, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 7820914, domain: 943850, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 20720355, domain: 2111503, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746336, user_id: 460953, query_id: 2868275, day: DayOfWeek::Mon, query_words: vec![1468238,924909,1777674,1603687], url: 55954880, domain: 4457118, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 47237060, domain: 4010128, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 23492758, domain: 2363287, relevance: ClickSat::Skip, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 14545097, domain: 1493148, relevance: ClickSat::Skip, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 53915363, domain: 4373273, relevance: ClickSat::Skip, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 26621811, domain: 2605392, relevance: ClickSat::Skip, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 14552093, domain: 1493218, relevance: ClickSat::Skip, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 35421594, domain: 3338000, relevance: ClickSat::Skip, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 48230874, domain: 4077200, relevance: ClickSat::Skip, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 14555484, domain: 1493360, relevance: ClickSat::High, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 14191798, day: DayOfWeek::Tue, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 31410893, domain: 3082034, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 46794187, domain: 3987448, relevance: ClickSat::Skip, position: Rank::from_usize(1), query_counter: 1, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 41458504, domain: 3689464, relevance: ClickSat::High, position: Rank::from_usize(2), query_counter: 1, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 39689963, domain: 3577505, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 1, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 70399115, domain: 5192758, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 1, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 3703011, domain: 490731, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 1, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 28829672, domain: 2833512, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 1, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 3307349, domain: 448064, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 1, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 46702262, domain: 3985421, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 1, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 63337004, domain: 4843018, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 1, },
            SearchResult { session_id: 2746337, user_id: 460953, query_id: 17296754, day: DayOfWeek::Tue, query_words: vec![4219157,2933841,2360314,323894], url: 48733046, domain: 4113642, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 1, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 53533848, domain: 4359008, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 62946524, domain: 4828867, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 62491081, domain: 4807815, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 32031240, domain: 3134365, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 39918555, domain: 3595774, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 18498300, domain: 1947616, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 43653204, domain: 3825236, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 40058986, domain: 3599848, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 56177047, domain: 4466491, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746338, user_id: 460953, query_id: 17181669, day: DayOfWeek::Tue, query_words: vec![4219157,2383044,1403353], url: 28836950, domain: 2833569, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 69827886, domain: 5157997, relevance: ClickSat::High, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 70943655, domain: 5267270, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 32129007, domain: 3140281, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 169532, domain: 37590, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 70512516, domain: 5211321, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 19556954, domain: 2021907, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 29103972, domain: 2854312, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 12336786, domain: 1233129, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 28310481, domain: 2790282, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746339, user_id: 460953, query_id: 17367682, day: DayOfWeek::Sun, query_words: vec![4219157,3424615,2383044,2870281,3514644], url: 33814388, domain: 3248872, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 31862626, domain: 3114734, relevance: ClickSat::Miss, position: Rank::from_usize(1), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 50628289, domain: 4217515, relevance: ClickSat::Miss, position: Rank::from_usize(2), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 19597275, domain: 2024884, relevance: ClickSat::Miss, position: Rank::from_usize(3), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 8983251, domain: 1048488, relevance: ClickSat::Miss, position: Rank::from_usize(4), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 33469474, domain: 3248439, relevance: ClickSat::Miss, position: Rank::from_usize(5), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 1847691, domain: 273345, relevance: ClickSat::Miss, position: Rank::from_usize(6), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 31181347, domain: 3050096, relevance: ClickSat::Miss, position: Rank::from_usize(7), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 54907932, domain: 4417726, relevance: ClickSat::Miss, position: Rank::from_usize(8), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 5297487, domain: 680403, relevance: ClickSat::Miss, position: Rank::from_usize(9), query_counter: 0, },
            SearchResult { session_id: 2746340, user_id: 460953, query_id: 18821770, day: DayOfWeek::Thu, query_words: vec![4390589,363714,3048434], url: 28238976, domain: 2780457, relevance: ClickSat::Miss, position: Rank::from_usize(10), query_counter: 0, },
        ];

        //auto generated
        #[rustfmt::skip]
        let current_search_results = &[
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 47237060, domain: 4010128, initial_rank: Rank::from_usize(1), query_counter: 0, },
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 23492758, domain: 2363287, initial_rank: Rank::from_usize(2), query_counter: 0, },
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 14545097, domain: 1493148, initial_rank: Rank::from_usize(3), query_counter: 0, },
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 53915363, domain: 4373273, initial_rank: Rank::from_usize(4), query_counter: 0, },
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 26621811, domain: 2605392, initial_rank: Rank::from_usize(5), query_counter: 0, },
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 35421594, domain: 3338000, initial_rank: Rank::from_usize(6), query_counter: 0, },
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 14552093, domain: 1493218, initial_rank: Rank::from_usize(7), query_counter: 0, },
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 14555484, domain: 1493360, initial_rank: Rank::from_usize(8), query_counter: 0, },
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 31410893, domain: 3082034, initial_rank: Rank::from_usize(9), query_counter: 0, },
            CurrentSearchResult { session_id: 2746341, user_id: 460953, query_id: 14191798, day: DayOfWeek::Thu, query_words: vec![3791236,2452511,3024236,3442492,3489517], url: 48230874, domain: 4077200, initial_rank: Rank::from_usize(10), query_counter: 0, },
        ];

        //auto generated
        #[rustfmt::skip]
        let features = &[
            [1.0, 0.283, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.6415, 0.0, 1.0, -1.0, 0.6415, 0.283, 0.283, 0.6415, 1.0, -1.0, 0.283, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.283, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 0.0, 0.0, 0.0, 5.0, 0.0],
            [2.0, 0.283, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.39149999999999996, 0.0, 1.0, -1.0, 0.39149999999999996, 0.283, 0.283, 0.39149999999999996, 1.0, -1.0, 0.283, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.283, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 1.0, 0.0, 0.0, 5.0, 0.0],
            [3.0, 0.283, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.30816666666666664, 0.0, 1.0, -1.0, 0.30816666666666664, 0.283, 0.283, 0.30816666666666664, 1.0, -1.0, 0.283, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.283, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 2.0, 0.0, 0.0, 5.0, 0.0],
            [4.0, 0.283, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.26649999999999996, 0.0, 1.0, -1.0, 0.26649999999999996, 0.283, 0.283, 0.26649999999999996, 1.0, -1.0, 0.283, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.283, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 3.0, 0.0, 0.0, 5.0, 0.0],
            [5.0, 0.283, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.2415, 0.0, 1.0, -1.0, 0.2415, 0.283, 0.283, 0.2415, 1.0, -1.0, 0.283, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.283, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 4.0, 0.0, 0.0, 5.0, 0.0],
            [6.0, 0.283, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.2129285714285714, 0.0, 1.0, -1.0, 0.2129285714285714, 0.283, 0.283, 0.2129285714285714, 1.0, -1.0, 0.283, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.283, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 5.0, 0.0, 0.0, 5.0, 0.0],
            [7.0, 0.283, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.22483333333333333, 0.0, 1.0, -1.0, 0.22483333333333333, 0.283, 0.283, 0.22483333333333333, 1.0, -1.0, 0.283, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.283, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 6.0, 0.0, 0.0, 5.0, 0.0],
            [8.0, 0.19705555555555554, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.19705555555555554, 1.0, 1.0, 0.0, 0.19705555555555554, 0.19705555555555554, 0.283, 0.283, 1.0, 0.0, 0.283, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.283, 0.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 7.0, 0.0, 0.0, 5.0, 1.25],
            [9.0, 0.283, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.1915, 0.0, 1.0, 0.0, 0.1915, 0.283, 0.1915, 0.283, 1.0, 0.0, 0.283, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.1915, 0.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 7.0, 0.0, 1.0, 5.0, 0.0],
            [10.0, 0.283, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.204, 0.0, 1.0, -1.0, 0.204, 0.283, 0.283, 0.204, 1.0, -1.0, 0.283, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.283, -1.0, 0.0, 5.0, 0.0, 1.0, 1.0, 0.19705555555555554, 1.0, 9.0, 0.9864267287308424, 8.0, 0.0, 1.0, 15.0, 3.933333333333333, 4.916666666666667, 7.0, 0.0, 1.0, 5.0, 0.0]
        ];

        let query = Query {
            id: current_search_results[0].query_id,
            words: current_search_results[0].query_words.clone(),
        };
        do_test_compute_features(history, &query, current_search_results, features);
    }
}
