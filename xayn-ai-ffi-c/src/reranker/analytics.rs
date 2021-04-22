use ffi_support::implement_into_ffi_by_pointer;
use xayn_ai::Analytics;

use crate::result::call_with_result;

/// The raw analytics.
pub struct CAnalytics(pub(crate) Option<Analytics>);

// this is more like a dummy impl for now until we have fleshed out the analytics
implement_into_ffi_by_pointer! { CAnalytics }

impl CAnalytics {
    /// See [`analytics_drop()`] for more.
    unsafe fn drop(analytics: *mut Self) {
        if !analytics.is_null() {
            unsafe { Box::from_raw(analytics) };
        }
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
pub unsafe extern "C" fn analytics_drop(analytics: *mut CAnalytics) {
    let drop = || {
        unsafe { CAnalytics::drop(analytics) };
        Ok(())
    };
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}
