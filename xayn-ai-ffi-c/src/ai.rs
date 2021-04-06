use std::{
    collections::HashMap,
    panic::{catch_unwind, RefUnwindSafe},
    ptr::null_mut,
    slice::from_raw_parts_mut,
};

use ffi_support::{call_with_result, implement_into_ffi_by_pointer, ExternError, FfiStr, IntoFfi};
use xayn_ai::{BetaSampler, Builder, DummyDatabase, Reranker, Systems};

use crate::{
    document::{CDocument, CHistory},
    error::CXaynAiError,
};

/// The Xayn AI.
///
/// # Examples
/// - Create a Xayn AI with [`xaynai_new()`].
/// - Rerank documents with [`xaynai_rerank()`].
/// - Free memory with [`xaynai_drop()`], [`ranks_drop()`] and [`error_message_drop()`].
///
/// [`error_message_drop()`]: crate::error::error_message_drop
pub struct CXaynAi(Reranker<Systems<DummyDatabase, BetaSampler>>);

impl RefUnwindSafe for CXaynAi {
    // safety: the field CXaynAi.0.errors must not be accessed after a panic; we don't access this
    // field anyways and there is no access to it for a caller of the ffi
}

implement_into_ffi_by_pointer! { CXaynAi }

/// Creates and initializes the Xayn AI.
///
/// # Errors
/// Returns a null pointer if:
/// - The vocab or model paths are invalid.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null vocab or model path doesn't point to an aligned, contiguous area of memory with a
/// terminating null byte.
/// - A non-null error doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_new(
    vocab: FfiStr,
    model: FfiStr,
    error: *mut ExternError,
) -> <CXaynAi as IntoFfi>::Value {
    let call = || {
        let vocab = vocab.as_opt_str().ok_or_else(|| {
            CXaynAiError::VocabPointer.with_context(
                "Failed to initialize the ai: The vocab is not a valid C-string pointer",
            )
        })?;
        let model = model.as_opt_str().ok_or_else(|| {
            CXaynAiError::ModelPointer.with_context(
                "Failed to initialize the ai: The model is not a valid C-string pointer",
            )
        })?;

        Builder::default()
            .with_database(DummyDatabase)
            .with_bert_from_file(vocab, model)
            .map_err(|cause| {
                CXaynAiError::ReadFile
                    .with_context(format!("Failed to initialize the ai: {}", cause))
            })?
            .build()
            .map(CXaynAi)
            .map_err(|cause| {
                CXaynAiError::InitAi.with_context(format!("Failed to initialize the ai: {}", cause))
            })
    };

    if let Some(error) = unsafe { error.as_mut() } {
        call_with_result(error, call)
    } else if let Ok(Ok(xaynai)) = catch_unwind(call) {
        xaynai.into_ffi_value()
    } else {
        CXaynAi::ffi_default()
    }
}

/// The reranked ranks of the documents.
///
/// The array is in the same order as the documents used in [`xaynai_rerank()`].
pub struct CRanks(Vec<u32>);

unsafe impl IntoFfi for CRanks {
    type Value = *mut u32;

    #[inline]
    fn ffi_default() -> Self::Value {
        null_mut()
    }

    #[inline]
    fn into_ffi_value(self) -> Self::Value {
        self.0.leak().as_mut_ptr()
    }
}

/// Reranks the documents with the Xayn AI.
///
/// # Errors
/// Returns a null pointer if:
/// - The xaynai is null.
/// - The document history is invalid.
/// - The documents are invalid.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null history array doesn't point to an aligned, contiguous area of memory with at least
/// history size many [`CHistory`]s.
/// - A history size is too large to address the memory of a non-null history array.
/// - A non-null documents array doesn't point to an aligned, contiguous area of memory with at
/// least documents size many [`CDocument`]s.
/// - A documents size is too large to address the memory of a non-null documents array.
/// - A non-null id or snippet doesn't point to an aligned, contiguous area of memory with a
/// terminating null byte.
/// - A non-null error doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
/// - A non-null, zero-sized ranks array is dereferenced.
#[no_mangle]
pub unsafe extern "C" fn xaynai_rerank(
    xaynai: <CXaynAi as IntoFfi>::Value,
    history: *const CHistory,
    history_size: u32,
    documents: *const CDocument,
    documents_size: u32,
    error: *mut ExternError,
) -> <CRanks as IntoFfi>::Value {
    let call = || {
        let xaynai = unsafe { xaynai.as_mut() }.ok_or_else(|| {
            CXaynAiError::AiPointer
                .with_context("Failed to rerank the documents: The ai pointer is null")
        })?;

        let history = if history_size == 0 {
            Vec::new()
        } else {
            unsafe {
                history
                    .as_ref()
                    .ok_or_else(|| {
                        CXaynAiError::HistoryPointer.with_context(
                            "Failed to rerank the documents: The document history pointer is null",
                        )
                    })?
                    .to_history(history_size)?
            }
        };

        let documents = if documents_size == 0 {
            Vec::new()
        } else {
            unsafe {
                documents
                    .as_ref()
                    .ok_or_else(|| {
                        CXaynAiError::DocumentsPointer.with_context(
                            "Failed to rerank the documents: The documents pointer is null",
                        )
                    })?
                    .to_documents(documents_size)?
            }
        };

        let ranks = xaynai
            .0
            .rerank(history.as_slice(), documents.as_slice())
            .into_iter()
            .map(|(id, rank)| (id, rank as u32))
            .collect::<HashMap<_, _>>();
        documents
            .iter()
            .map(|document| ranks.get(&document.id).copied())
            .collect::<Option<Vec<_>>>()
            .map(CRanks)
            .ok_or_else(|| {
                CXaynAiError::Internal.with_context(
                    "Failed to rerank the documents: The document ids are inconsistent",
                )
            })
    };

    if let Some(error) = unsafe { error.as_mut() } {
        call_with_result(error, call)
    } else if let Ok(Ok(ranks)) = catch_unwind(call) {
        ranks.into_ffi_value()
    } else {
        CRanks::ffi_default()
    }
}

/// Frees the memory of the Xayn AI.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null xaynai is freed more than once.
/// - A non-null xaynai is accessed after being freed.
#[no_mangle]
pub unsafe extern "C" fn xaynai_drop(xaynai: <CXaynAi as IntoFfi>::Value) {
    let _ = catch_unwind(|| {
        if !xaynai.is_null() {
            unsafe { Box::from_raw(xaynai) };
        }
    });
}

/// Frees the memory of the ranks array.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null ranks doesn't point to memory allocated by [`xaynai_rerank()`].
/// - A non-zero ranks size is different from the documents size used in [`xaynai_rerank()`].
/// - A non-null ranks is freed more than once.
/// - A non-null ranks is accessed after being freed.
#[no_mangle]
pub unsafe extern "C" fn ranks_drop(ranks: <CRanks as IntoFfi>::Value, ranks_size: u32) {
    let _ = catch_unwind(|| {
        if ranks_size > 0 && !ranks.is_null() {
            unsafe { Box::from_raw(from_raw_parts_mut(ranks, ranks_size as usize)) };
        }
    });
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, ptr::null};

    use super::*;
    use crate::utils::tests::{drop_values, setup_pointers, setup_values};

    #[test]
    fn test_rerank_full() {
        let (vocab, model, hist, hist_size, docs, docs_size, mut error) = setup_values();
        let (c_vocab, c_model, c_hist, c_docs, c_error) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let xaynai = unsafe { xaynai_new(c_vocab, c_model, c_error) };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CXaynAiError::Success);

        let ranks = unsafe {
            xaynai_rerank(
                xaynai,
                c_hist.as_ptr(),
                hist_size,
                c_docs.as_ptr(),
                docs_size,
                c_error,
            )
        };
        assert_eq!(error.get_code(), CXaynAiError::Success);

        unsafe { xaynai_drop(xaynai) };
        unsafe { ranks_drop(ranks, docs_size) };
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_rerank_empty() {
        let (vocab, model, hist, _, docs, _, mut error) = setup_values();
        let (c_vocab, c_model, _, _, c_error) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let xaynai = unsafe { xaynai_new(c_vocab, c_model, c_error) };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CXaynAiError::Success);

        let c_hist = null();
        let hist_size = 0;
        let c_docs = null();
        let docs_size = 0;
        let ranks = unsafe { xaynai_rerank(xaynai, c_hist, hist_size, c_docs, docs_size, c_error) };
        assert_eq!(error.get_code(), CXaynAiError::Success);

        unsafe { xaynai_drop(xaynai) };
        unsafe { ranks_drop(ranks, docs_size) };
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_rerank_history_empty() {
        let (vocab, model, hist, _, docs, docs_size, mut error) = setup_values();
        let (c_vocab, c_model, _, c_docs, c_error) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let xaynai = unsafe { xaynai_new(c_vocab, c_model, c_error) };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CXaynAiError::Success);

        let c_hist = null();
        let hist_size = 0;
        let ranks = unsafe {
            xaynai_rerank(
                xaynai,
                c_hist,
                hist_size,
                c_docs.as_ptr(),
                docs_size,
                c_error,
            )
        };
        assert_eq!(error.get_code(), CXaynAiError::Success);

        unsafe { xaynai_drop(xaynai) };
        unsafe { ranks_drop(ranks, docs_size) };
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_rerank_documents_empty() {
        let (vocab, model, hist, hist_size, docs, _, mut error) = setup_values();
        let (c_vocab, c_model, c_hist, _, c_error) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let xaynai = unsafe { xaynai_new(c_vocab, c_model, c_error) };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CXaynAiError::Success);

        let c_docs = null();
        let docs_size = 0;
        let ranks = unsafe {
            xaynai_rerank(
                xaynai,
                c_hist.as_ptr(),
                hist_size,
                c_docs,
                docs_size,
                c_error,
            )
        };
        assert_eq!(error.get_code(), CXaynAiError::Success);

        unsafe { xaynai_drop(xaynai) };
        unsafe { ranks_drop(ranks, docs_size) };
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_vocab_null() {
        let (vocab, model, hist, _, docs, _, mut error) = setup_values();
        let (_, c_model, _, _, c_error) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let c_invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(unsafe { xaynai_new(c_invalid, c_model, c_error) }.is_null());
        assert_eq!(error.get_code(), CXaynAiError::VocabPointer);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: The vocab is not a valid C-string pointer",
        );

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_vocab_invalid() {
        let (vocab, model, hist, _, docs, _, mut error) = setup_values();
        let (_, c_model, _, _, c_error) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let invalid = CString::new("").unwrap();
        let c_invalid = FfiStr::from_cstr(invalid.as_c_str());
        assert!(unsafe { xaynai_new(c_invalid, c_model, c_error) }.is_null());
        assert_eq!(error.get_code(), CXaynAiError::ReadFile);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: Failed to load a data file: No such file or directory (os error 2)",
        );

        invalid.into_string().unwrap();
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_model_null() {
        let (vocab, model, hist, _, docs, _, mut error) = setup_values();
        let (c_vocab, _, _, _, c_error) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let c_invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(unsafe { xaynai_new(c_vocab, c_invalid, c_error) }.is_null());
        assert_eq!(error.get_code(), CXaynAiError::ModelPointer);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: The model is not a valid C-string pointer",
        );

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_model_invalid() {
        let (vocab, model, hist, _, docs, _, mut error) = setup_values();
        let (c_vocab, _, _, _, c_error) = setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let invalid = CString::new("").unwrap();
        let c_invalid = FfiStr::from_cstr(invalid.as_c_str());
        assert!(unsafe { xaynai_new(c_vocab, c_invalid, c_error) }.is_null());
        assert_eq!(error.get_code(), CXaynAiError::ReadFile);
        assert_eq!(
            error.get_message(),
            "Failed to initialize the ai: Failed to load a data file: No such file or directory (os error 2)",
        );

        invalid.into_string().unwrap();
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_ai_null() {
        let (vocab, model, hist, hist_size, docs, docs_size, mut error) = setup_values();
        let (_, _, c_hist, mut c_docs, c_error) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let c_invalid = null_mut();
        assert!(unsafe {
            xaynai_rerank(
                c_invalid,
                c_hist.as_ptr(),
                hist_size,
                c_docs.as_mut_ptr(),
                docs_size,
                c_error,
            )
        }
        .is_null());
        assert_eq!(error.get_code(), CXaynAiError::AiPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The ai pointer is null",
        );

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_history_null() {
        let (vocab, model, hist, hist_size, docs, docs_size, mut error) = setup_values();
        let (c_vocab, c_model, _, mut c_docs, c_error) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let xaynai = unsafe { xaynai_new(c_vocab, c_model, c_error) };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CXaynAiError::Success);

        let c_invalid = null();
        assert!(unsafe {
            xaynai_rerank(
                xaynai,
                c_invalid,
                hist_size,
                c_docs.as_mut_ptr(),
                docs_size,
                c_error,
            )
        }
        .is_null());
        assert_eq!(error.get_code(), CXaynAiError::HistoryPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The document history pointer is null",
        );

        unsafe { xaynai_drop(xaynai) };
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_documents_null() {
        let (vocab, model, hist, hist_size, docs, docs_size, mut error) = setup_values();
        let (c_vocab, c_model, c_hist, _, c_error) =
            setup_pointers(&vocab, &model, &hist, &docs, &mut error);

        let xaynai = unsafe { xaynai_new(c_vocab, c_model, c_error) };
        assert!(!xaynai.is_null());
        assert_eq!(error.get_code(), CXaynAiError::Success);

        let c_invalid = null_mut();
        assert!(unsafe {
            xaynai_rerank(
                xaynai,
                c_hist.as_ptr(),
                hist_size,
                c_invalid,
                docs_size,
                c_error,
            )
        }
        .is_null());
        assert_eq!(error.get_code(), CXaynAiError::DocumentsPointer);
        assert_eq!(
            error.get_message(),
            "Failed to rerank the documents: The documents pointer is null",
        );

        unsafe { xaynai_drop(xaynai) };
        drop_values(vocab, model, hist, docs, error);
    }
}
