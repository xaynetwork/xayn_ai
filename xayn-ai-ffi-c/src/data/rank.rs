use crate::{slice::CBoxedSlice, utils::IntoRaw};

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
pub type CRanks = CBoxedSlice<u32>;

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
        Some(Box::new(self.0.into_boxed_slice().into()))
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
pub unsafe extern "C" fn ranks_drop(_ranks: Option<Box<CRanks>>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_raw() {
        let buffer = (0..10).collect::<Vec<_>>();
        let ranks = Ranks(buffer.clone()).into_raw().unwrap();
        assert_eq!(ranks.as_slice(), buffer);
    }

    #[test]
    fn test_into_empty() {
        let ranks = Ranks(Vec::new()).into_raw().unwrap();
        assert!(ranks.is_empty());
    }
}
