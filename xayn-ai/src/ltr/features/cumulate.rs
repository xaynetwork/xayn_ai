#![allow(dead_code)] // TEMP

use super::{
    cond_prob,
    AtomFeat,
    FeatMap,
    FilterPred,
    HistSearchResult,
    NewSearchResult,
    UrlOrDom,
};
use std::collections::HashMap;

/// Cumulated features for a given user.
pub(crate) struct CumFeatures {
    /// Cumulated feature for matching URL.
    url: FeatMap,
}

impl CumFeatures {
    /// Builds the cumulated features for a given search result.
    ///
    /// These are given by sums of conditional probabilities:
    /// ```text
    /// sum{cond_prob(outcome, pred(r.url))}
    /// ```
    /// where the sum ranges over each search result `r` ranked above `res`. `pred` is the predicate
    /// corresponding to the cumulated feature, and `outcome` one of its specified atoms.
    pub(crate) fn build(hist: &[HistSearchResult], res: impl AsRef<NewSearchResult>) -> Self {
        let res = res.as_ref();
        let url = hist
            .iter()
            // if res is ranked n, get the n-1 results ranked above res
            .filter(|r| r.query == res.query && r.rerank < res.init_rank)
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

        Self { url }
    }
}
