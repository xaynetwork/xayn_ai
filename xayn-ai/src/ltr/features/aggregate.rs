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
pub(crate) struct AggregFeatures {
    /// Aggregate feature for matching domain.
    pub(crate) dom: FeatMap,
    /// Aggregate feature for matching domain over anterior sessions.
    pub(crate) dom_ant: FeatMap,
    /// Aggregate feature for matching URL.
    pub(crate) url: FeatMap,
    /// Aggregate feature for matching URL over anterior sessions.
    pub(crate) url_ant: FeatMap,
    /// Aggregate feature for matching domain and query.
    pub(crate) dom_query: FeatMap,
    /// Aggregate feature for matching domain and query over anterior sessions.
    pub(crate) dom_query_ant: FeatMap,
    /// Aggregate feature for matching URL and query.
    pub(crate) url_query: FeatMap,
    /// Aggregate feature for matching URL and query over anterior sessions.
    pub(crate) url_query_ant: FeatMap,
    /// Aggregate feature for matching URL and query over current session.
    pub(crate) url_query_curr: FeatMap,
}

impl AggregFeatures {
    /// Build aggregate features for the given search result and history of a user.
    pub(crate) fn build(hist: &[HistSearchResult], doc: impl AsRef<DocSearchResult>) -> Self {
        let doc = doc.as_ref();

        let anterior = SessionCond::Anterior(doc.query.session_id);
        let current = SessionCond::Current(doc.query.session_id);
        let doc_url = UrlOrDom::Url(&doc.url);
        let doc_dom = UrlOrDom::Dom(&doc.domain);

        let pred_dom = FilterPred::new(doc_dom);
        let dom = aggreg_feat(hist, pred_dom);
        let dom_ant = aggreg_feat(hist, pred_dom.with_session(anterior));

        let pred_url = FilterPred::new(doc_url);
        let url = aggreg_feat(hist, pred_url);
        let url_ant = aggreg_feat(hist, pred_url.with_session(anterior));

        let pred_dom_query = pred_dom.with_query(doc.query.query_id);
        let dom_query = aggreg_feat(hist, pred_dom_query);
        let dom_query_ant = aggreg_feat(hist, pred_dom_query.with_session(anterior));

        let pred_url_query = pred_url.with_query(doc.query.query_id);
        let url_query = aggreg_feat(hist, pred_url_query);
        let url_query_ant = aggreg_feat(hist, pred_url_query.with_session(anterior));
        let url_query_curr = aggreg_feat(hist, pred_url_query.with_session(current));

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

fn aggreg_feat(hist: &[HistSearchResult], pred: FilterPred) -> FeatMap {
    let eval_atom = |atom_feat| match atom_feat {
        AtomFeat::MeanRecipRank(outcome) => mean_recip_rank(hist, Some(outcome), Some(pred)),
        AtomFeat::MeanRecipRankAll => mean_recip_rank(hist, None, Some(pred)),
        AtomFeat::SnippetQuality => snippet_quality(hist, pred),
        AtomFeat::CondProb(outcome) => cond_prob(hist, outcome, pred),
    };
    pred.agg_atoms()
        .into_iter()
        .map(|atom_feat| (atom_feat, eval_atom(atom_feat)))
        .collect()
}
