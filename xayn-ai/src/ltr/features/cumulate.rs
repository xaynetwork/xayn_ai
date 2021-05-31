#![allow(dead_code)] // TEMP

use crate::ltr::features::dataiku::{cond_prob, FilterPred, SearchResult, UrlOrDom};

use super::dataiku::{ClickSat, CurrentSearchResult};

/// Cumulated features for a given user.
#[derive(Clone)]
pub(super) struct CumFeatures {
    // Cumulative "score" for matching skipped results.
    pub(super) skip: f32,
    // Cumulative "score" for matching clicked (medium/not last clicked) results.
    pub(super) medium: f32,
    // Cumulative "score" for matching clicked (heigh/last clicked) results.
    pub(super) high: f32,
}

/// Accumulator to compute cumulative features.
///
/// After creating it [`Self.extract_next()`] needs to be called in ascending order of the
/// initial ranking (starting from `Rank::First`).
pub(super) struct CumFeaturesAccumulator {
    //FIXME do we need statistical correct floating point multiplication? (I don't think so tbh.)
    //      (I.e. keep a list of results for skip,medium,heigh and sort? it
    //       in asc order before summing to reduce rounding errors)
    next_features: CumFeatures,
}

impl CumFeaturesAccumulator {
    /// Creates a new "empty" accumulator for extracting cumulative features.
    pub(super) fn new() -> Self {
        Self {
            next_features: CumFeatures {
                skip: 0.0,
                medium: 0.0,
                high: 0.0,
            }
        }
    }

    /// Extracts cumulative features for given current search results.
    ///
    /// Must be called in ascending ranking order for all current results in the
    /// list of results for the current search query.
    pub(super) fn extract_next(
        &mut self,
        history: &[SearchResult],
        current_result: &CurrentSearchResult,
    ) -> CumFeatures {
        let features = self.next_features.clone();

        let url_filter = FilterPred::new(UrlOrDom::Url(current_result.url));
        self.next_features.skip += cond_prob(history, ClickSat::Skip, url_filter);
        self.next_features.medium += cond_prob(history, ClickSat::Medium, url_filter);
        self.next_features.high += cond_prob(history, ClickSat::High, url_filter);

        features
    }
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

        // skip - medium - high
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

        for offset in 0..=2 {
            let mut acc = CumFeaturesAccumulator::new();

            for (idx, expected) in expected_results.iter().enumerate() {
                let mut current = CurrentSearchResult::from(&history[offset * 10 + idx]);
                // Test that changes in the session do not affect the outcome.
                // (It should only filter by URL.)
                current.session_id += (offset % 2) as i32;
                let features = acc.extract_next(&history, &current);

                assert_approx_eq!(f32, features.skip, expected[0]);
                assert_approx_eq!(f32, features.medium, expected[1]);
                assert_approx_eq!(f32, features.high, expected[2]);
            }
        }
    }
}
