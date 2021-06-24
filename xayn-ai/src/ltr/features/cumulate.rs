use super::{cond_prob, Action, DocSearchResult, FilterPred, HistSearchResult, UrlOrDom};

#[derive(Clone)]
/// Cumulative features.
pub(super) struct CumulatedFeatures {
    /// Cumulative skip score for matching URLs.
    pub(super) url_skip: f32,
    /// Cumulative click1 score for matching URLs.
    pub(super) url_click1: f32,
    /// Cumulative click2 score for matching URLs.
    pub(super) url_click2: f32,
}

/// Accumulator to compute cumulative features.
///
/// After creating it [`Self.build_next()`] needs to be called in ascending order of the
/// initial ranking (starting from `Rank::First`).
pub(super) struct CumFeaturesAccumulator {
    next_features: CumulatedFeatures,
}

impl CumFeaturesAccumulator {
    /// Creates a new "empty" accumulator for building cumulative features.
    pub(super) fn new() -> Self {
        Self {
            next_features: CumulatedFeatures {
                url_skip: 0.,
                url_click1: 0.,
                url_click2: 0.,
            },
        }
    }

    /// Builds cumulative features for the given search results.
    ///
    /// Must be called in ascending ranking order for all results of the
    /// current search query.
    pub(super) fn build_next(
        &mut self,
        hists: &[HistSearchResult],
        doc: &DocSearchResult,
    ) -> CumulatedFeatures {
        let features = self.next_features.clone();

        let url_pred = FilterPred::new(UrlOrDom::Url(&doc.url));
        self.next_features.url_skip += cond_prob(hists, Action::Skip, url_pred);
        self.next_features.url_click1 += cond_prob(hists, Action::Click1, url_pred);
        self.next_features.url_click2 += cond_prob(hists, Action::Click2, url_pred);

        features
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::mock_uuid;

    use super::super::{Action, DayOfWeek, Query, QueryId, Rank, SessionId};

    use super::*;

    fn history<'a>(iter: impl IntoIterator<Item = &'a (&'a str, Action)>) -> Vec<HistSearchResult> {
        iter.into_iter()
            .enumerate()
            .map(|(id, (domain, action))| {
                let in_query_id = id % 10;
                let per_query_id = id / 10;
                HistSearchResult {
                    query: Query {
                        session_id: SessionId(mock_uuid(1)),
                        query_count: per_query_id,
                        query_id: QueryId(mock_uuid(per_query_id)),
                        query_words: vec![per_query_id.to_string()],
                    },
                    url: in_query_id.to_string(),
                    domain: domain.to_string(),
                    final_rank: Rank(in_query_id),
                    day: DayOfWeek::Tue,
                    action: *action,
                }
            })
            .collect()
    }

    fn create_mock_search_result(hist: &HistSearchResult) -> DocSearchResult {
        DocSearchResult {
            query: hist.query.clone(),
            url: hist.url.clone(),
            domain: hist.domain.clone(),
            initial_rank: hist.final_rank,
        }
    }

    #[test]
    fn test_cum_features() {
        let history = history(&[
            /* query 1 */
            ("1", Action::Skip),
            ("2", Action::Skip),
            ("3", Action::Click1),
            ("3", Action::Click2),
            ("4", Action::Miss),
            ("5", Action::Miss),
            ("6", Action::Miss),
            ("7", Action::Miss),
            ("8", Action::Miss),
            ("9", Action::Miss),
            /* query 2 */
            ("1", Action::Skip),
            ("2", Action::Skip),
            ("3", Action::Skip),
            ("3", Action::Skip),
            ("4", Action::Skip),
            ("5", Action::Skip),
            ("6", Action::Click1),
            ("7", Action::Click1),
            ("8", Action::Click2),
            ("9", Action::Miss),
            /* query 3 */
            ("1", Action::Skip),
            ("2", Action::Click1),
            ("3", Action::Click1),
            ("3", Action::Click1),
            ("4", Action::Click2),
            ("5", Action::Miss),
            ("6", Action::Miss),
            ("7", Action::Miss),
            ("8", Action::Miss),
            ("9", Action::Miss),
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
                let mut current = create_mock_search_result(&history[offset * 10 + idx]);
                // Test that changes in the session do not affect the outcome.
                // (It should only filter by URL.)
                if offset % 2 == 0 {
                    current.query.session_id = SessionId(mock_uuid(123534123));
                }
                let features = acc.build_next(&history, &current);

                assert_approx_eq!(f32, features.url_skip, expected[0]);
                assert_approx_eq!(f32, features.url_click1, expected[1]);
                assert_approx_eq!(f32, features.url_click2, expected[2]);
            }
        }
    }
}
