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
            assert_approx_eq!(f32, feature_row[45], cum_features.url[skipped]);
            assert_approx_eq!(f32, feature_row[46], cum_features.url[click1]);
            assert_approx_eq!(f32, feature_row[47], cum_features.url[click2]);

            assert_approx_eq!(f32, feature_row[48], terms_variety as f32, ulps = 0);

            let seasonality = seasonality(history, search_result.domain);
            assert_approx_eq!(f32, feature_row[49], seasonality);
        }
    }

    #[test]
    fn test_full_training() {
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
}
