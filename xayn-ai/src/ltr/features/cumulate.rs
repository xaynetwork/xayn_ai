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
