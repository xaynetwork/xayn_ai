use std::{collections::HashMap, slice};

use ffi_support::{
    abort_on_panic::{call_with_result, with_abort_on_panic},
    implement_into_ffi_by_pointer,
    ErrorCode,
    ExternError,
    FfiStr,
};
use rubert::{AveragePooler, Builder as BertBuilder};
use xayn_ai::{
    BetaSampler,
    CoiConfiguration,
    CoiSystem,
    ConstLtr,
    Context,
    DummyAnalytics,
    MabRanking,
    Reranker,
};

use crate::{
    document::{docs_to_vec, hist_to_vec, CDocument, CHistory},
    error::CXaynAiError,
    systems::{DummyDatabase, Systems},
};

/// The Xayn AI.
///
/// # Examples
/// - Create a Xayn AI with [`xaynai_new()`].
/// - Rerank documents with [`xaynai_rerank()`].
/// - Free memory with [`xaynai_drop()`] and [`error_message_drop()`].
///
/// [`error_message_drop()`]: crate::error::error_message_drop
pub struct CXaynAi(Reranker<Systems>);

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
) -> *mut CXaynAi {
    unsafe fn call(vocab: FfiStr, model: FfiStr, error: &mut ExternError) -> *mut CXaynAi {
        call_with_result(error, || {
            let vocab = vocab.as_opt_str().ok_or_else(|| {
                ExternError::new_error(
                    ErrorCode::new(CXaynAiError::VocabPointer as i32),
                    "Failed to build the bert model: The vocab is not a valid C-string pointer",
                )
            })?;
            let model = model.as_opt_str().ok_or_else(|| {
                ExternError::new_error(
                    ErrorCode::new(CXaynAiError::ModelPointer as i32),
                    "Failed to build the bert model: The model is not a valid C-string pointer",
                )
            })?;

            let bert = BertBuilder::from_files(vocab, model)
                .map_err(|cause| {
                    ExternError::new_error(
                        ErrorCode::new(CXaynAiError::ReadFile as i32),
                        format!("Failed to build the bert model: {}", cause),
                    )
                })?
                .with_token_size(90)
                .expect("infallible: token size >= 2")
                .with_accents(false)
                .with_lowercase(true)
                .with_pooling(AveragePooler)
                .build()
                .map_err(|cause| {
                    ExternError::new_error(
                        ErrorCode::new(CXaynAiError::BuildBert as i32),
                        format!("Failed to build the bert model: {}", cause),
                    )
                })?;

            let coi = CoiSystem::new(CoiConfiguration::default());
            let ltr = ConstLtr(0.5);
            let context = Context;
            let mab = MabRanking::new(BetaSampler);

            // TODO: use the reranker builder once it is available
            let systems = Systems {
                // TODO: use the actual database once it is available
                database: DummyDatabase,
                bert,
                coi,
                ltr,
                context,
                mab,
                // TODO: use the actual analytics once it is available
                analytics: DummyAnalytics,
            };
            Reranker::new(systems).map(CXaynAi).map_err(|cause| {
                ExternError::new_error(
                    ErrorCode::new(CXaynAiError::BuildReranker as i32),
                    format!("Failed to build the reranker: {}", cause),
                )
            })
        })
    }

    if let Some(error) = unsafe { error.as_mut() } {
        unsafe { call(vocab, model, error) }
    } else {
        let mut error = ExternError::default();
        let xaynai = unsafe { call(vocab, model, &mut error) };
        unsafe { error.manually_release() };
        xaynai
    }
}

/// Reranks the documents with the Xayn AI.
///
/// The reranked order is written to the ranks of the documents array.
///
/// # Errors
/// Returns without changing the ranks if:
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
#[no_mangle]
pub unsafe extern "C" fn xaynai_rerank(
    xaynai: *mut CXaynAi,
    hist: *const CHistory,
    hist_size: u32,
    docs: *mut CDocument,
    docs_size: u32,
    error: *mut ExternError,
) {
    unsafe fn call(
        xaynai: *mut CXaynAi,
        hist: *const CHistory,
        hist_size: u32,
        docs: *mut CDocument,
        docs_size: u32,
        error: &mut ExternError,
    ) {
        call_with_result(error, || {
            let xaynai = unsafe { xaynai.as_mut() }.ok_or_else(|| {
                ExternError::new_error(
                    ErrorCode::new(CXaynAiError::XaynAiPointer as i32),
                    "Failed to rerank the documents: The xaynai pointer is null",
                )
            })?;

            let hist = if hist.is_null() {
                return Err(ExternError::new_error(
                    ErrorCode::new(CXaynAiError::HistoryPointer as i32),
                    "Failed to rerank the documents: The document history pointer is null",
                ));
            } else if hist_size == 0 {
                &[]
            } else {
                unsafe { slice::from_raw_parts(hist, hist_size as usize) }
            };
            let history = hist_to_vec(hist)?;

            let docs = if docs.is_null() {
                return Err(ExternError::new_error(
                    ErrorCode::new(CXaynAiError::DocumentsPointer as i32),
                    "Failed to rerank the documents: The documents pointer is null",
                ));
            } else if docs_size == 0 {
                &mut []
            } else {
                unsafe { slice::from_raw_parts_mut(docs, docs_size as usize) }
            };
            let documents = docs_to_vec(docs)?;

            let ranks = xaynai
                .0
                .rerank(history.as_slice(), documents.as_slice())
                .into_iter()
                .collect::<HashMap<_, _>>();
            for (doc, document) in docs.iter_mut().zip(documents) {
                doc.rank = ranks[&document.id] as u32;
            }

            Ok(())
        })
    }

    if let Some(error) = unsafe { error.as_mut() } {
        unsafe { call(xaynai, hist, hist_size, docs, docs_size, error) }
    } else {
        let mut error = ExternError::default();
        let xaynai = unsafe { call(xaynai, hist, hist_size, docs, docs_size, &mut error) };
        unsafe { error.manually_release() };
        xaynai
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
pub unsafe extern "C" fn xaynai_drop(xaynai: *mut CXaynAi) {
    with_abort_on_panic(|| {
        if !xaynai.is_null() {
            unsafe { Box::from_raw(xaynai) };
        }
    })
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::CString,
        ptr::{null, null_mut},
    };

    use super::*;
    use crate::{
        error::tests::{error_code, error_message},
        utils::tests::{drop_values, setup_pointers, setup_values},
    };

    #[test]
    fn test_rerank() {
        let (vocab, model, hist, hist_size, docs, docs_size, error) = setup_values();
        let (c_vocab, c_model, c_hist, mut c_docs, error) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            hist.as_slice(),
            docs.as_slice(),
            error,
        );

        let xaynai = unsafe { xaynai_new(c_vocab, c_model, error) };
        assert!(!xaynai.is_null());
        assert_eq!(error_code(error), CXaynAiError::Success);

        unsafe {
            xaynai_rerank(
                xaynai,
                c_hist.as_ptr(),
                hist_size,
                c_docs.as_mut_ptr(),
                docs_size,
                error,
            )
        };
        assert_eq!(error_code(error), CXaynAiError::Success);

        unsafe { xaynai_drop(xaynai) };
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_vocab_null() {
        let (vocab, model, hist, _, docs, _, error) = setup_values();
        let (_, c_model, _, _, error) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            hist.as_slice(),
            docs.as_slice(),
            error,
        );

        let c_invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(unsafe { xaynai_new(c_invalid, c_model, error) }.is_null());
        assert_eq!(error_code(error), CXaynAiError::VocabPointer);
        assert_eq!(
            error_message(error),
            "Failed to build the bert model: The vocab is not a valid C-string pointer",
        );

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_vocab_invalid() {
        let (vocab, model, hist, _, docs, _, error) = setup_values();
        let (_, c_model, _, _, error) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            hist.as_slice(),
            docs.as_slice(),
            error,
        );

        let invalid = CString::new("").unwrap();
        let c_invalid = FfiStr::from_cstr(invalid.as_c_str());
        assert!(unsafe { xaynai_new(c_invalid, c_model, error) }.is_null());
        assert_eq!(error_code(error), CXaynAiError::ReadFile);
        assert_eq!(
            error_message(error),
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)",
        );

        invalid.into_string().unwrap();
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_model_null() {
        let (vocab, model, hist, _, docs, _, error) = setup_values();
        let (c_vocab, _, _, _, error) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            hist.as_slice(),
            docs.as_slice(),
            error,
        );

        let c_invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(unsafe { xaynai_new(c_vocab, c_invalid, error) }.is_null());
        assert_eq!(error_code(error), CXaynAiError::ModelPointer);
        assert_eq!(
            error_message(error),
            "Failed to build the bert model: The model is not a valid C-string pointer",
        );

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_model_invalid() {
        let (vocab, model, hist, _, docs, _, error) = setup_values();
        let (c_vocab, _, _, _, error) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            hist.as_slice(),
            docs.as_slice(),
            error,
        );

        let invalid = CString::new("").unwrap();
        let c_invalid = FfiStr::from_cstr(invalid.as_c_str());
        assert!(unsafe { xaynai_new(c_vocab, c_invalid, error) }.is_null());
        assert_eq!(error_code(error), CXaynAiError::ReadFile);
        assert_eq!(
            error_message(error),
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)",
        );

        invalid.into_string().unwrap();
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_ai_null() {
        let (vocab, model, hist, hist_size, docs, docs_size, error) = setup_values();
        let (_, _, c_hist, mut c_docs, error) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            hist.as_slice(),
            docs.as_slice(),
            error,
        );

        unsafe {
            xaynai_rerank(
                null_mut(),
                c_hist.as_ptr(),
                hist_size,
                c_docs.as_mut_ptr(),
                docs_size,
                error,
            )
        };
        assert_eq!(error_code(error), CXaynAiError::XaynAiPointer);
        assert_eq!(
            error_message(error),
            "Failed to rerank the documents: The xaynai pointer is null",
        );

        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_history_null() {
        let (vocab, model, hist, hist_size, docs, docs_size, error) = setup_values();
        let (c_vocab, c_model, _, mut c_docs, error) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            hist.as_slice(),
            docs.as_slice(),
            error,
        );

        let xaynai = unsafe { xaynai_new(c_vocab, c_model, error) };
        let c_invalid = null();
        unsafe {
            xaynai_rerank(
                xaynai,
                c_invalid,
                hist_size,
                c_docs.as_mut_ptr(),
                docs_size,
                error,
            )
        };
        assert_eq!(error_code(error), CXaynAiError::HistoryPointer);
        assert_eq!(
            error_message(error),
            "Failed to rerank the documents: The document history pointer is null",
        );

        unsafe { xaynai_drop(xaynai) };
        drop_values(vocab, model, hist, docs, error);
    }

    #[test]
    fn test_documents_null() {
        let (vocab, model, hist, hist_size, docs, docs_size, error) = setup_values();
        let (c_vocab, c_model, c_hist, _, error) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            hist.as_slice(),
            docs.as_slice(),
            error,
        );

        let xaynai = unsafe { xaynai_new(c_vocab, c_model, error) };
        let c_invalid = null_mut();
        unsafe {
            xaynai_rerank(
                xaynai,
                c_hist.as_ptr(),
                hist_size,
                c_invalid,
                docs_size,
                error,
            )
        };
        assert_eq!(error_code(error), CXaynAiError::DocumentsPointer);
        assert_eq!(
            error_message(error),
            "Failed to rerank the documents: The documents pointer is null",
        );

        unsafe { xaynai_drop(xaynai) };
        drop_values(vocab, model, hist, docs, error);
    }
}
