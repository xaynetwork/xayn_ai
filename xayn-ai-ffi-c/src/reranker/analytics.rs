use crate::utils::IntoRaw;

/// The analytics of the reranker.
pub struct Analytics(pub(crate) Option<xayn_ai::Analytics>);

#[repr(C)]
pub struct CAnalytics {
    /// The nDCG@k score between the LTR ranking and the relevance based ranking
    pub ndcg_ltr: f32,
    /// The nDCG@k score between the Context ranking and the relevance based ranking
    pub ndcg_context: f32,
    /// The nDCG@k score between the initial ranking and the relevance based ranking
    pub ndcg_initial_ranking: f32,
    /// The nDCG@k score between the final ranking and the relevance based ranking
    pub ndcg_final_ranking: f32,
}

// this is more like a dummy impl for now until we have fleshed out the analytics
unsafe impl IntoRaw for Analytics
where
    CAnalytics: Sized,
{
    // Safety:
    // CAnalytics is sized, hence Box<CAnalytics> is representable as a *mut CAnalytics and
    // Option<Box<CAnalytics>> is eligible for the nullable pointer optimization.
    type Value = Option<Box<CAnalytics>>;

    #[inline]
    fn into_raw(self) -> Self::Value {
        self.0.map(|analytics| {
            let xayn_ai::Analytics {
                ndcg_ltr,
                ndcg_context,
                ndcg_initial_ranking,
                ndcg_final_ranking,
            } = analytics;
            Box::new(CAnalytics {
                ndcg_ltr,
                ndcg_context,
                ndcg_initial_ranking,
                ndcg_final_ranking,
            })
        })
    }
}

/// Frees the memory of the analytics.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `analytics` doesn't point to memory allocated by [`xaynai_analytics()`].
/// - A non-null `analytics` is freed more than once.
/// - A non-null `analytics` is accessed after being freed.
///
/// [`xaynai_analytics()`]: crate::reranker::ai::xaynai_analytics
#[no_mangle]
pub unsafe extern "C" fn analytics_drop(_analytics: Option<Box<CAnalytics>>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tests::AsPtr;

    impl AsPtr for CAnalytics {}
}
