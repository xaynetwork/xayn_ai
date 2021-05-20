#![allow(dead_code)] // TEMP

use crate::ltr::features::dataiku::{
    cond_prob,
    AtomFeat,
    FeatMap,
    FilterPred,
    SearchResult,
    UrlOrDom,
};
use std::collections::HashMap;

/// Cumulated features for a given user.
pub(crate) struct CumFeatures {
    /// Cumulated feature for matching URL.
    url: FeatMap,
}

/// Determines the cumulated features for a given search result.
///
/// These are given by sums of conditional probabilities:
/// ```text
/// sum{cond_prob(outcome, pred(r.url))}
/// ```
/// where the sum ranges over each search result `r` ranked above `res`. `pred` is the predicate
/// corresponding to the cumulated feature, and `outcome` one of its specified atoms.
pub(crate) fn cum_features(hist: &[SearchResult], res: impl AsRef<SearchResult>) -> CumFeatures {
    let res = res.as_ref();
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
            let pred = FilterPred::new(UrlOrDom::Url(&r.url));
            pred.cum_atoms()
                .into_iter()
                .map(move |outcome| (outcome, cond_prob(hist, outcome, pred)))
        })
        // sum cond probs for each outcome
        .fold(HashMap::new(), |mut cp_map, (outcome, cp)| {
            *cp_map.entry(AtomFeat::CondProb(outcome)).or_default() += cp;
            cp_map
        });

    CumFeatures { url }
}
