use std::{
    collections::HashMap,
    ptr::{null, null_mut},
    slice::from_raw_parts_mut,
};

use ffi_support::{ExternError, IntoFfi};
use xayn_ai::{Document, DocumentsRank};

use crate::result::{call_with_result, error::CError};

/// The ranks of the reranked documents.
pub struct Ranks(Vec<u32>);

/// A raw slice of ranks.
///
/// The ranks are in the same logical order as the documents used in [`xaynai_rerank()`].
///
/// [`xaynai_rerank()`]: crate::ai::xaynai_rerank
#[repr(C)]
pub struct CRanks {
    /// The raw pointer to the ranks.
    pub data: *const u32,
    /// The number of ranks.
    pub len: u32,
}

impl Ranks {
    /// Reorders the ranks wrt. the documents.
    pub fn from_reranked_documents(
        ranks: DocumentsRank,
        documents: &[Document],
    ) -> Result<Self, ExternError> {
        let ranks = ranks
            .into_iter()
            .map(|(id, rank)| (id, rank as u32))
            .collect::<HashMap<_, _>>();
        documents
            .iter()
            .map(|document| ranks.get(&document.id).copied())
            .collect::<Option<Vec<_>>>()
            .map(Self)
            .ok_or_else(|| {
                CError::Internal.with_context(
                    "Failed to rerank the documents: The document ids are inconsistent",
                )
            })
    }
}

unsafe impl IntoFfi for Ranks {
    type Value = *mut CRanks;

    #[inline]
    fn ffi_default() -> Self::Value {
        null_mut()
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        let len = self.0.len() as u32;
        let data = if self.0.is_empty() {
            null()
        } else {
            self.0.leak().as_ptr()
        };
        let ranks = CRanks { data, len };

        Box::into_raw(Box::new(ranks))
    }
}

impl CRanks {
    /// See [`ranks_drop()`] for more.
    unsafe fn drop(ranks: *mut Self) {
        if !ranks.is_null() {
            let ranks = unsafe { Box::from_raw(ranks) };
            if !ranks.data.is_null() && ranks.len > 0 {
                unsafe {
                    Box::from_raw(from_raw_parts_mut(
                        ranks.data as *mut u32,
                        ranks.len as usize,
                    ))
                };
            }
        }
    }
}

/// Frees the memory of the ranks array.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null `ranks` doesn't point to memory allocated by [`xaynai_rerank()`].
/// - A non-zero `len` is different from the documents `len` used in [`xaynai_rerank()`].
/// - A non-null `ranks` is freed more than once.
/// - A non-null `ranks` is accessed after being freed.
///
/// [`xaynai_rerank()`]: crate::ai::xaynai_rerank
#[no_mangle]
pub unsafe extern "C" fn ranks_drop(ranks: *mut CRanks) {
    let drop = || {
        unsafe { CRanks::drop(ranks) };
        Result::<_, ExternError>::Ok(())
    };
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
        let ranks = Ranks(buffer.clone()).into_ffi_value();

        assert!(!ranks.is_null());
        let data = unsafe { &*ranks }.data;
        let len = unsafe { &*ranks }.len as usize;
        assert!(!data.is_null());
        assert_eq!(len, buffer.len());
        assert_eq!(unsafe { from_raw_parts(data, len) }, buffer);

        unsafe { ranks_drop(ranks) };
    }

    #[test]
    fn test_into_empty() {
        let ranks = Ranks(Vec::new()).into_ffi_value();

        assert!(!ranks.is_null());
        assert!(unsafe { &*ranks }.data.is_null());
        assert_eq!(unsafe { &*ranks }.len, 0);

        unsafe { ranks_drop(ranks) };
    }
}
