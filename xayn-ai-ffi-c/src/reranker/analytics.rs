use crate::utils::IntoRaw;

/// The analytics of the reranker.
pub(super) struct Analytics(pub(crate) Option<xayn_ai::Analytics>);

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
            Box::new(CAnalytics {
                ndcg_ltr: analytics.ndcg_ltr,
                ndcg_context: analytics.ndcg_context,
                ndcg_initial_ranking: analytics.ndcg_initial_ranking,
                ndcg_final_ranking: analytics.ndcg_final_ranking,
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
    use xayn_ai::assert_f32_eq;

    use super::*;

    #[test]
    fn test_convert_some_analytics_to_c_analytics() {
        let analytics = Analytics(Some(xayn_ai::Analytics {
            ndcg_ltr: 0.25,
            ndcg_context: 0.75,
            ndcg_initial_ranking: 1.125,
            ndcg_final_ranking: 2.825,
        }));

        let c_analytics = analytics.into_raw().unwrap();

        assert_f32_eq!(c_analytics.ndcg_ltr, 0.25, ulps = 0);
        assert_f32_eq!(c_analytics.ndcg_context, 0.75, ulps = 0);
        assert_f32_eq!(c_analytics.ndcg_initial_ranking, 1.125, ulps = 0);
        assert_f32_eq!(c_analytics.ndcg_final_ranking, 2.825, ulps = 0);

        unsafe {
            analytics_drop(Some(c_analytics));
        }
    }

    #[test]
    fn test_convert_none_analytics_to_c_analytics() {
        let analytics = Analytics(None);
        let c_analytics = analytics.into_raw();

        assert!(c_analytics.is_none());

        unsafe {
            analytics_drop(c_analytics);
        }
    }
}
