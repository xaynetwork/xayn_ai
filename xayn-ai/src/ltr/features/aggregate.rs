#![allow(dead_code)] // TEMP

use crate::ltr::features::dataiku::{
    cond_prob,
    mean_recip_rank,
    snippet_quality,
    AtomFeat,
    FeatMap,
    FilterPred,
    SearchResult,
    SessionCond,
    UrlOrDom,
};

/// Aggregate features for a given user.
//FIXME[comment/resolve with or after review]: fields
//  We know exact which features are run so why is this half a struct with explicit fields and
//  half a struct using dynamic field mapping?
struct AggregateFeatures {
    /// Aggregate feature for matching domain.
    dom: FeatMap,
    /// Aggregate feature for matching domain over anterior sessions.
    dom_ant: FeatMap,
    /// Aggregate feature for matching URL.
    url: FeatMap,
    /// Aggregate feature for matching URL over anterior sessions.
    url_ant: FeatMap,
    /// Aggregate feature for matching domain and query.
    dom_query: FeatMap,
    /// Aggregate feature for matching domain and query over anterior sessions.
    dom_query_ant: FeatMap,
    /// Aggregate feature for matching URL and query.
    url_query: FeatMap,
    /// Aggregate feature for matching URL and query over anterior sessions.
    url_query_ant: FeatMap,
    /// Aggregate feature for matching URL and query over current session.
    url_query_curr: FeatMap,
}

/// Calculate aggregate features for the given search result and history of a user.
fn aggregate_features(hist: &[SearchResult], r: SearchResult) -> AggregateFeatures {
    let anterior = SessionCond::Anterior(r.session_id);
    let current = SessionCond::Current(r.session_id);
    let r_url = UrlOrDom::Url(r.url);
    let r_dom = UrlOrDom::Dom(r.domain);

    let pred_dom = FilterPred::new(r_dom);
    let dom = aggregate_feature(hist, pred_dom);
    let dom_ant = aggregate_feature(hist, pred_dom.with_session(anterior));

    let pred_url = FilterPred::new(r_url);
    let url = aggregate_feature(hist, pred_url);
    let url_ant = aggregate_feature(hist, pred_url.with_session(anterior));

    let pred_dom_query = pred_dom.with_query(r.query_id);
    let dom_query = aggregate_feature(hist, pred_dom_query);
    let dom_query_ant = aggregate_feature(hist, pred_dom_query.with_session(anterior));

    let pred_url_query = pred_url.with_query(r.query_id);
    let url_query = aggregate_feature(hist, pred_url_query);
    let url_query_ant = aggregate_feature(hist, pred_url_query.with_session(anterior));
    let url_query_curr = aggregate_feature(hist, pred_url_query.with_session(current));

    AggregateFeatures {
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

//FIXME this isn't calculating a single aggregate feature but a feature group.
fn aggregate_feature(hist: &[SearchResult], pred: FilterPred) -> FeatMap {
    //FIXME why not implement this on AtomFeat, like atom_feat.evaluate(hist, pred)
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

#[cfg(test)]
mod tests {
    use crate::ltr::features::dataiku::{ClickSat, MrrOutcome};

    use super::*;

    #[test]
    fn test_aggregate_feature() {
        // this just tests if the mapping from filter => atoms => functionality works
        let history = &[];
        let filter = FilterPred::new(UrlOrDom::Url(1))
            .with_query(2)
            .with_session(SessionCond::Anterior(10));
        let map = aggregate_feature(history, filter);

        assert_approx_eq!(f32, map[&AtomFeat::MeanRecipRankAll], 0.283);
        assert_approx_eq!(f32, map[&AtomFeat::MeanRecipRank(MrrOutcome::Click)], 0.283);
        assert_approx_eq!(f32, map[&AtomFeat::MeanRecipRank(MrrOutcome::Miss)], 0.283);
        assert_approx_eq!(f32, map[&AtomFeat::MeanRecipRank(MrrOutcome::Skip)], 0.283);
        assert_approx_eq!(f32, map[&AtomFeat::SnippetQuality], 0.0);
        assert_approx_eq!(f32, map[&AtomFeat::CondProb(ClickSat::Miss)], 0.0);

        assert_eq!(map.len(), 6);
    }

    #[ignore = "needs big history blob"]
    #[test]
    fn test_aggregate_features() {
        todo!()
    }
}
