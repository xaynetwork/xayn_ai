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

use super::dataiku::CurrentSearchResult;

/// Cumulated features for a given user.
pub(super) struct CumFeatures {
    /// Cumulated feature for matching URL.
    //FIXME: Why is this using a map when this are literally only 3 values??
    pub(super) url: FeatMap,
}

impl CumFeatures {
    pub(super) fn extract(hist: &[SearchResult], res: &CurrentSearchResult) -> CumFeatures {
        //FIXME temp. to make reviews easier by not showing the whole `cum_features` function as changed
        cum_features(hist, res)
    }
}

/// Determines the cumulated features for a given search result.
///
/// These are given by sums of conditional probabilities:
/// ```text
/// sum{cond_prob(outcome, pred(r.url))}
/// ```
/// where the sum ranges over each search result `r` ranked above `res`. `pred` is the predicate
/// corresponding to the cumulated feature, and `outcome` one of its specified atoms.
fn cum_features(hist: &[SearchResult], res: &CurrentSearchResult) -> CumFeatures {
    let mut url = hist
        .iter()
        //FIXME The current query is normally not in hist, as such doing this
        //      with non test data with not work ass the specific combination of
        //      (res.session_id, res.initial_rank) won't appear in the history
        //TODO  First make sure this fails a test.
        // if res is ranked n, get the n-1 results ranked above res
        .filter(|r| {
            // this is filtered by session and query but no such filter is done in python I think
            // same is true for < position,  what is used in python is the filter by URL
            r.session_id == res.session_id
                && r.query_id == res.query_id
                && r.query_counter == res.query_counter
                && r.position < res.initial_rank
        })
        // calculate specified cond probs for each of the above
        .flat_map(|r| {
            //FIXME this is the only place we ever call cum_atoms with only that specific predicate
            //      we maybe should change how we handle this.
            let pred = FilterPred::new(UrlOrDom::Url(r.url));
            pred.cum_atoms()
                .into_iter()
                .map(move |outcome| (outcome, cond_prob(hist, outcome, pred)))
        })
        // sum cond probs for each outcome
        .fold(HashMap::new(), |mut cp_map, (outcome, cp)| {
            *cp_map.entry(AtomFeat::CondProb(outcome)).or_default() += cp;
            cp_map
        });

    if url.is_empty() {
        //FIXME this is not the best solution but I don't want to touch
        //      aboves code in this commit.
        for atom in FilterPred::new(UrlOrDom::Url(0)).cum_atoms().into_iter() {
            url.insert(AtomFeat::CondProb(atom), 0.0);
        }
    }

    CumFeatures { url }
}

#[cfg(test)]
mod tests {
    use crate::ltr::features::dataiku::{ClickSat, DayOfWeek, Rank};

    use super::*;

    fn history<'a>(iter: impl IntoIterator<Item = &'a (i32, ClickSat)>) -> Vec<SearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (domain, relevance))| {
                let id = id as i32;
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                SearchResult {
                    session_id: 1,
                    user_id: 1,
                    query_id: per_query_id,
                    day: DayOfWeek::Tue,
                    query_words: vec![1, 2, id],
                    url: in_query_id,
                    domain: *domain,
                    relevance: *relevance,
                    position: Rank::from_usize(1 + in_query_id as usize),
                    query_counter: per_query_id as u8,
                }
            })
            .collect()
    }

    #[test]
    fn test_cum_features() {
        let history = history(&[
            /* query 1 */
            (1, ClickSat::Skip),
            (2, ClickSat::Skip),
            (3, ClickSat::Medium),
            (3, ClickSat::High),
            (4, ClickSat::Miss),
            (5, ClickSat::Miss),
            (6, ClickSat::Miss),
            (7, ClickSat::Miss),
            (8, ClickSat::Miss),
            (9, ClickSat::Miss),
            /* query 2 */
            (1, ClickSat::Skip),
            (2, ClickSat::Skip),
            (3, ClickSat::Skip),
            (3, ClickSat::Skip),
            (4, ClickSat::Skip),
            (5, ClickSat::Skip),
            (6, ClickSat::Medium),
            (7, ClickSat::Medium),
            (8, ClickSat::High),
            (9, ClickSat::Miss),
            /* query 3 */
            (1, ClickSat::Skip),
            (2, ClickSat::Medium),
            (3, ClickSat::Medium),
            (3, ClickSat::Medium),
            (4, ClickSat::High),
            (5, ClickSat::Miss),
            (6, ClickSat::Miss),
            (7, ClickSat::Miss),
            (8, ClickSat::Miss),
            (9, ClickSat::Miss),
        ]);

        let expected_results = &[
            [0., 0., 0.],
            [1., 0., 0.],
            [1. + 2. / 3., 1. / 3., 0.],
            [2., 1., 0.],
            [2. + 1. / 3., 1. + 1. / 3., 1. / 3.],
            [2. + 2. / 3., 1. + 1. / 3., 2. / 3.],
            [3., 1. + 1. / 3., 2. / 3.],
            [3., 1. + 2. / 3., 2. / 3.],
            [3., 2., 2. / 3.],
            [3., 2., 1.],
        ];

        for offset in &[0, 20] {
            for (idx, expected) in expected_results.iter().enumerate() {
                let map = cum_features(&history, &history[*offset + idx].clone().into()).url;
                let values = [
                    map[&AtomFeat::CondProb(ClickSat::Skip)],
                    map[&AtomFeat::CondProb(ClickSat::Medium)],
                    map[&AtomFeat::CondProb(ClickSat::High)],
                ];
                assert_approx_eq!(f32, values, expected);
                assert_eq!(map.len(), 3);
            }
        }
    }
}
