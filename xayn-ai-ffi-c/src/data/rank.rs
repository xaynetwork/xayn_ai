use std::{panic::AssertUnwindSafe, slice::from_raw_parts_mut};

use ffi_support::{ExternError, IntoFfi};

use crate::result::call_with_result;

/// The ranks of the reranked documents.
pub struct Ranks(Vec<u32>);

impl From<xayn_ai::Ranks> for Ranks {
    fn from(ranks: xayn_ai::Ranks) -> Self {
        Self(ranks.into_iter().map(|rank| rank as u32).collect())
    }
}

/// A raw slice of ranks.
///
/// The ranks are in the same logical order as the documents used in [`xaynai_rerank()`].
///
/// [`xaynai_rerank()`]: crate::reranker::ai::xaynai_rerank
#[repr(C)]
pub struct CRanks<'a> {
    /// The raw pointer to the ranks.
    pub data: Option<&'a u32>,
    /// The number of ranks.
    pub len: u32,
}

unsafe impl IntoFfi for Ranks {
    type Value = Option<&'static mut CRanks<'static>>;

    #[inline]
    fn ffi_default() -> Self::Value {
        None
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        let len = self.0.len() as u32;
        let data = if self.0.is_empty() {
            None
        } else {
            self.0.leak().first()
        };

        Some(Box::leak(Box::new(CRanks { data, len })))
    }
}

impl CRanks<'_> {
    /// See [`ranks_drop()`] for more.
    unsafe fn drop(ranks: Option<&mut Self>) {
        if let Some(ranks) = ranks {
            let ranks = unsafe { Box::from_raw(ranks) };
            if let Some(data) = ranks.data {
                if ranks.len > 0 {
                    unsafe {
                        Box::from_raw(from_raw_parts_mut(
                            data as *const u32 as *mut u32,
                            ranks.len as usize,
                        ))
                    };
                }
            }
        }
    }
}

/// Frees the memory of the ranks.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `ranks` doesn't point to memory allocated by [`xaynai_rerank()`].
/// - A non-null `ranks` is freed more than once.
/// - A non-null `ranks` is accessed after being freed.
///
/// [`xaynai_rerank()`]: crate::reranker::ai::xaynai_rerank
#[no_mangle]
pub unsafe extern "C" fn ranks_drop(ranks: Option<&mut CRanks>) {
    let drop = AssertUnwindSafe(|| {
        unsafe { CRanks::drop(ranks) };
        Result::<_, ExternError>::Ok(())
    });
    let clean = || {};
    let error = None;

    call_with_result(drop, clean, error);
}

#[cfg(test)]
pub(crate) mod tests {
    use std::slice::from_raw_parts;

    use super::*;

    #[test]
    fn test_into_raw() {
        let buffer = (0..10).collect::<Vec<_>>();
        let ranks = Ranks(buffer.clone()).into_ffi_value().unwrap();

        assert!(ranks.data.is_some());
        assert_eq!(ranks.len as usize, buffer.len());
        assert_eq!(
            unsafe { from_raw_parts(ranks.data.unwrap(), ranks.len as usize) },
            buffer,
        );

        unsafe { ranks_drop(Some(ranks)) };
    }

    #[test]
    fn test_into_empty() {
        let ranks = Ranks(Vec::new()).into_ffi_value().unwrap();

        assert!(ranks.data.is_none());
        assert_eq!(ranks.len, 0);

        unsafe { ranks_drop(Some(ranks)) };
    }
}
