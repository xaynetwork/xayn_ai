use std::{convert::Infallible, panic::AssertUnwindSafe, slice};

use crate::{result::call_with_result, utils::IntoRaw};

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
pub struct CRanks {
    /// The raw pointer to the ranks.
    pub data: Option<Box<u32>>,
    /// The number of ranks.
    pub len: u32,
}

unsafe impl IntoRaw for Ranks
where
    CRanks: Sized,
{
    // Safety:
    // CRanks is sized, hence Box<CRanks> is representable as a *mut CRanks and Option<Box<CRanks>>
    // is eligible for the nullable pointer optimization.
    type Value = Option<Box<CRanks>>;

    #[inline]
    fn into_raw(self) -> Self::Value {
        let len = self.0.len() as u32;
        let data = if self.0.is_empty() {
            None
        } else {
            // Safety:
            // Casting a Box<[u32]> to a Box<u32> is sound, but it leaks all values except the very
            // first one. Hence we store the length of the slice next to the pointer to be able to
            // reclaim the memory.
            Some(unsafe { Box::from_raw(self.0.leak().as_mut_ptr()) })
        };

        Some(Box::new(CRanks { data, len }))
    }
}

impl CRanks {
    /// See [`ranks_drop()`] for more.
    #[allow(clippy::unnecessary_wraps)]
    unsafe fn drop(ranks: Option<Box<Self>>) -> Result<(), Infallible> {
        if let Some(ranks) = ranks {
            if let Some(data) = ranks.data {
                if ranks.len > 0 {
                    // Safety:
                    // Casting a Box<u32> to a Box<[u32]> is sound, if it originated from a boxed
                    // slice with corresponding length.
                    unsafe {
                        Box::from_raw(slice::from_raw_parts_mut(
                            Box::into_raw(data),
                            ranks.len as usize,
                        ))
                    };
                }
            }
        }

        Ok(())
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
pub unsafe extern "C" fn ranks_drop(ranks: Option<Box<CRanks>>) {
    let drop = AssertUnwindSafe(
        // Safety: The memory is dropped anyways.
        || unsafe { CRanks::drop(ranks) },
    );
    let error = None;

    call_with_result(drop, error);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tests::AsPtr;

    impl AsPtr for CRanks {}

    #[test]
    fn test_into_raw() {
        let buffer = (0..10).collect::<Vec<_>>();
        let ranks = Ranks(buffer.clone()).into_raw().unwrap();

        let data = ranks.data.as_ref().unwrap().as_ref();
        let len = ranks.len as usize;
        assert_eq!(len, buffer.len());
        assert_eq!(unsafe { slice::from_raw_parts(data, len) }, buffer);

        unsafe { ranks_drop(ranks.into_ptr()) };
    }

    #[test]
    fn test_into_empty() {
        let ranks = Ranks(Vec::new()).into_raw().unwrap();

        assert!(ranks.data.is_none());
        assert_eq!(ranks.len, 0);

        unsafe { ranks_drop(ranks.into_ptr()) };
    }
}
