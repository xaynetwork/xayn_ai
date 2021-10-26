use super::{
    cond_prob,
    mean_recip_rank,
    snippet_quality,
    AtomFeat,
    DocSearchResult,
    FeatMap,
    FilterPred,
    HistSearchResult,
    SessionCond,
    UrlOrDom,
};

/// Aggregate features for a given user.
pub(super) struct AggregateFeatures {
    /// Aggregate feature for matching domain.
    pub(super) dom: FeatMap,
    /// Aggregate feature for matching domain over anterior sessions.
    pub(super) dom_ant: FeatMap,
    /// Aggregate feature for matching URL.
    pub(super) url: FeatMap,
    /// Aggregate feature for matching URL over anterior sessions.
    pub(super) url_ant: FeatMap,
    /// Aggregate feature for matching domain and query.
    pub(super) dom_query: FeatMap,
    /// Aggregate feature for matching domain and query over anterior sessions.
    pub(super) dom_query_ant: FeatMap,
    /// Aggregate feature for matching URL and query.
    pub(super) url_query: FeatMap,
    /// Aggregate feature for matching URL and query over anterior sessions.
    pub(super) url_query_ant: FeatMap,
    /// Aggregate feature for matching URL and query over current session.
    pub(super) url_query_curr: FeatMap,
}

impl AggregateFeatures {
    /// Build aggregate features for the given search result and history of a user.
    pub(super) fn build(hists: &[HistSearchResult], doc: impl AsRef<DocSearchResult>) -> Self {
        let doc = doc.as_ref();

        let not_current = SessionCond::Anterior(doc.query.session_id);
        let current = SessionCond::Current(doc.query.session_id);
        let doc_url = UrlOrDom::Url(&doc.url);
        let doc_dom = UrlOrDom::Dom(&doc.domain);

        let pred_dom = FilterPred::new(doc_dom);
        let dom = aggregate_feature(hists, pred_dom);
        let dom_ant = aggregate_feature(hists, pred_dom.with_session(not_current));

        let pred_url = FilterPred::new(doc_url);
        let url = aggregate_feature(hists, pred_url);
        let url_ant = aggregate_feature(hists, pred_url.with_session(not_current));

        let pred_dom_query = pred_dom.with_query(doc.query.query_id);
        let dom_query = aggregate_feature(hists, pred_dom_query);
        let dom_query_ant = aggregate_feature(hists, pred_dom_query.with_session(not_current));

        let pred_url_query = pred_url.with_query(doc.query.query_id);
        let url_query = aggregate_feature(hists, pred_url_query);
        let url_query_ant = aggregate_feature(hists, pred_url_query.with_session(not_current));
        let url_query_curr = aggregate_feature(hists, pred_url_query.with_session(current));

        Self {
            dom,
            dom_ant,
            url,
            url_ant,
            dom_query,
            dom_query_ant,
            url_query,
            url_query_ant,
            url_query_curr,
        }
    }
}

fn aggregate_feature(hists: &[HistSearchResult], pred: FilterPred) -> FeatMap {
    let eval_atom = |atom_feat| match atom_feat {
        AtomFeat::MeanRecipRank(outcome) => mean_recip_rank(hists, Some(outcome), Some(pred)),
        AtomFeat::MeanRecipRankAll => mean_recip_rank(hists, None, Some(pred)),
        AtomFeat::SnippetQuality => snippet_quality(hists, pred),
        AtomFeat::CondProb(outcome) => cond_prob(hists, outcome, pred),
    };
    pred.agg_atoms()
        .into_iter()
        .map(|atom_feat| (atom_feat, eval_atom(atom_feat)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        super::{Action, MrrOutcome, QueryId, SessionId},
        *,
    };
    use crate::tests::mock_uuid;
    use test_utils::assert_approx_eq;

    #[test]
    fn test_aggregate_feature() {
        // This just tests if the mapping from filter => atoms => functionality works
        // all other parts are already tested in parent module.
        let history = &[];
        let filter = FilterPred::new(UrlOrDom::Url("1"))
            .with_query(QueryId(mock_uuid(2)))
            .with_session(SessionCond::Anterior(SessionId(mock_uuid(10))));
        let map = aggregate_feature(history, filter);

        assert_approx_eq!(f32, map[&AtomFeat::MeanRecipRankAll], 0.283);
        assert_approx_eq!(f32, map[&AtomFeat::MeanRecipRank(MrrOutcome::Click)], 0.283);
        assert_approx_eq!(f32, map[&AtomFeat::MeanRecipRank(MrrOutcome::Miss)], 0.283);
        assert_approx_eq!(f32, map[&AtomFeat::MeanRecipRank(MrrOutcome::Skip)], 0.283);
        assert_approx_eq!(f32, map[&AtomFeat::SnippetQuality], 0.0);
        assert_approx_eq!(f32, map[&AtomFeat::CondProb(Action::Miss)], 0.0);

        assert_eq!(map.len(), 6);
    }
}
