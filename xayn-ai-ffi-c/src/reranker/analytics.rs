use std::panic::AssertUnwindSafe;

use ffi_support::IntoFfi;
use xayn_ai::Analytics;

use crate::result::call_with_result;

/// The raw analytics.
pub struct CAnalytics(pub(crate) Option<Analytics>);

// this is more like a dummy impl for now until we have fleshed out the analytics
unsafe impl IntoFfi for CAnalytics {
    type Value = Option<&'static mut CAnalytics>;

    #[inline]
    fn ffi_default() -> Self::Value {
        None
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        Some(Box::leak(Box::new(self)))
    }
}

impl CAnalytics {
    /// See [`analytics_drop()`] for more.
    unsafe fn drop(analytics: Option<&mut Self>) {
        if let Some(analytics) = analytics {
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
pub unsafe extern "C" fn analytics_drop(analytics: Option<&mut CAnalytics>) {
    let drop = AssertUnwindSafe(|| {
        unsafe { CAnalytics::drop(analytics) };
        Ok(())
    });
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}
