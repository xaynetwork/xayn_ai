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
    Document,
    DummyAnalytics,
    MabRanking,
    Reranker,
};

use crate::{
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

/// A raw document.
#[repr(C)]
pub struct CDocument<'a> {
    /// The raw pointer to the document id.
    pub id: FfiStr<'a>,
    /// The raw pointer to the document snippet.
    pub snippet: FfiStr<'a>,
    /// The rank of the document.
    pub rank: u32,
}

/// Reranks the documents with the Xayn AI.
///
/// The reranked order is written to the ranks of the documents array.
///
/// # Errors
/// Returns without changing the ranks if:
/// - The xaynai is null.
/// - The documents are invalid.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null documents array doesn't point to an aligned, contiguous area of memory with
/// at least size [`CDocument`]s.
/// - A documents size is too large to address the memory of a non-null documents array.
/// - A non-null id or snippet doesn't point to an aligned, contiguous area of memory with a
/// terminating null byte.
/// - A non-null error doesn't point to an aligned, contiguous area of memory with an
/// [`ExternError`].
#[no_mangle]
pub unsafe extern "C" fn xaynai_rerank(
    xaynai: *const CXaynAi,
    docs: *mut CDocument,
    size: u32,
    error: *mut ExternError,
) {
    unsafe fn call(
        xaynai: *const CXaynAi,
        docs: *mut CDocument,
        size: u32,
        error: &mut ExternError,
    ) {
        call_with_result(error, || {
            let _xaynai = unsafe { xaynai.as_ref() }.ok_or_else(|| {
                ExternError::new_error(
                    ErrorCode::new(CXaynAiError::XaynAiPointer as i32),
                    "Failed to rerank the documents: The xaynai pointer is null",
                )
            })?;

            let docs = if docs.is_null() {
                return Err(ExternError::new_error(
                    ErrorCode::new(CXaynAiError::DocumentsPointer as i32),
                    "Failed to rerank the documents: The documents pointer is null",
                ));
            } else if size == 0 {
                &mut []
            } else {
                unsafe { slice::from_raw_parts_mut(docs, size as usize) }
            };
            let documents = docs.iter()
                .map(|document| {
                    let id = document
                        .id
                        .as_opt_str()
                        .map(Into::into)
                        .ok_or_else(|| ExternError::new_error(
                            ErrorCode::new(CXaynAiError::IdPointer as i32),
                            "Failed to rerank the documents: A document id is not a valid C-string pointer",
                        ))?;
                    let snippet = document
                        .snippet
                        .as_opt_str()
                        .map(Into::into)
                        .ok_or_else(|| ExternError::new_error(
                            ErrorCode::new(CXaynAiError::SnippetPointer as i32),
                            "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
                        ))?;
                    let rank = document.rank as usize;

                    Ok(Document { id, snippet, rank })
                })
                .collect::<Result<Vec<Document>, ExternError>>()?;

            // TODO: use the actual reranker once it is available
            let reranks = documents
                .iter()
                .map(|document| document.id.clone())
                .zip(documents.iter().map(|document| document.rank).rev())
                .collect::<HashMap<_, _>>();
            for (doc, document) in docs.iter_mut().zip(documents) {
                doc.rank = reranks[&document.id] as u32;
            }

            Ok(())
        })
    }

    if let Some(error) = unsafe { error.as_mut() } {
        unsafe { call(xaynai, docs, size, error) }
    } else {
        let mut error = ExternError::default();
        let xaynai = unsafe { call(xaynai, docs, size, &mut error) };
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
        ffi::{CStr, CString},
        mem::take,
        ptr::{null, null_mut},
    };

    use super::*;
    use crate::tests::{MODEL, VOCAB};

    /// Creates values for testing.
    #[allow(clippy::type_complexity)]
    fn setup_values() -> (
        CString,
        CString,
        Vec<(CString, CString, u32)>,
        u32,
        ExternError,
    ) {
        let vocab = CString::new(VOCAB).unwrap();
        let model = CString::new(MODEL).unwrap();

        let size = 10;
        let docs = (0..size)
            .map(|idx| {
                let id = CString::new(format!("{}", idx)).unwrap();
                let snippet = CString::new(format!("snippet {}", idx)).unwrap();
                let rank = idx;
                (id, snippet, rank)
            })
            .collect::<Vec<_>>();

        let error = ExternError::default();

        (vocab, model, docs, size, error)
    }

    /// Creates pointers for testing.
    fn setup_pointers<'a>(
        vocab: &'a CStr,
        model: &'a CStr,
        docs: &'a [(CString, CString, u32)],
        error: ExternError,
    ) -> (FfiStr<'a>, FfiStr<'a>, Vec<CDocument<'a>>, *mut ExternError) {
        let vocab = FfiStr::from_cstr(vocab);
        let model = FfiStr::from_cstr(model);

        let docs = docs
            .iter()
            .map(|(id, snippet, rank)| CDocument {
                id: FfiStr::from_cstr(id),
                snippet: FfiStr::from_cstr(snippet),
                rank: *rank,
            })
            .collect();

        let error = Box::into_raw(Box::new(error));

        (vocab, model, docs, error)
    }

    unsafe fn error_code(error: *const ExternError) -> i32 {
        unsafe { &*error }.get_code().code()
    }

    unsafe fn error_message(error: *mut ExternError) -> String {
        unsafe { take(&mut *error).get_and_consume_message() }.unwrap()
    }

    impl PartialEq<CXaynAiError> for i32 {
        fn eq(&self, other: &CXaynAiError) -> bool {
            *self == *other as i32
        }
    }

    #[test]
    fn test_xaynai_error() {
        assert_eq!(ErrorCode::PANIC.code(), CXaynAiError::Panic);
        assert_eq!(ErrorCode::SUCCESS.code(), CXaynAiError::Success);
    }

    #[test]
    fn test_xaynai_rerank() {
        let (vocab, model, docs, size, error) = setup_values();
        let (vocab, model, mut docs, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);

        let xaynai = unsafe { xaynai_new(vocab, model, error) };
        assert!(!xaynai.is_null());
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::Success);

        let reranks = docs.iter().map(|doc| doc.rank).rev().collect::<Vec<_>>();
        unsafe { xaynai_rerank(xaynai, docs.as_mut_ptr(), size, error) };
        assert!(docs.iter().map(|doc| doc.rank).eq(reranks));
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::Success);

        unsafe { xaynai_drop(xaynai) };
        unsafe { Box::from_raw(error) };
    }

    #[test]
    fn test_xaynai_vocab_null() {
        let (vocab, model, docs, _, error) = setup_values();
        let (_, model, _, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);

        let invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(unsafe { xaynai_new(invalid, model, error) }.is_null());
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::VocabPointer);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to build the bert model: The vocab is not a valid C-string pointer",
        );

        unsafe { Box::from_raw(error) };
    }

    #[test]
    fn test_xaynai_vocab_invalid() {
        let (vocab, model, docs, _, error) = setup_values();
        let (_, model, _, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);

        let invalid = CString::new("").unwrap();
        let invalid = FfiStr::from_cstr(invalid.as_c_str());
        assert!(unsafe { xaynai_new(invalid, model, error) }.is_null());
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::ReadFile);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)",
        );

        unsafe { Box::from_raw(error) };
    }

    #[test]
    fn test_xaynai_model_null() {
        let (vocab, model, docs, _, error) = setup_values();
        let (vocab, _, _, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);

        let invalid = unsafe { FfiStr::from_raw(null()) };
        assert!(unsafe { xaynai_new(vocab, invalid, error) }.is_null());
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::ModelPointer);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to build the bert model: The model is not a valid C-string pointer",
        );

        unsafe { Box::from_raw(error) };
    }

    #[test]
    fn test_xaynai_model_invalid() {
        let (vocab, model, docs, _, error) = setup_values();
        let (vocab, _, _, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);

        let invalid = CString::new("").unwrap();
        let invalid = FfiStr::from_cstr(invalid.as_c_str());
        assert!(unsafe { xaynai_new(vocab, invalid, error) }.is_null());
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::ReadFile);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)",
        );

        unsafe { Box::from_raw(error) };
    }

    #[test]
    fn test_xaynai_documents_null() {
        let (vocab, model, docs, size, error) = setup_values();
        let (vocab, model, _, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);
        let xaynai = unsafe { xaynai_new(vocab, model, error) };

        unsafe { xaynai_rerank(xaynai, null_mut(), size, error) };
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::DocumentsPointer);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to rerank the documents: The documents pointer is null",
        );

        unsafe { xaynai_drop(xaynai) };
        unsafe { Box::from_raw(error) };
    }

    #[test]
    fn test_xaynai_document_id_null() {
        let (vocab, model, docs, _, error) = setup_values();
        let (vocab, model, mut docs, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);
        let xaynai = unsafe { xaynai_new(vocab, model, error) };

        let mut invalid = vec![CDocument {
            id: unsafe { FfiStr::from_raw(null()) },
            snippet: docs.remove(0).snippet,
            rank: docs[0].rank,
        }];
        let invalid = invalid.as_mut_ptr();
        unsafe { xaynai_rerank(xaynai, invalid, 1, error) };
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::IdPointer);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to rerank the documents: A document id is not a valid C-string pointer",
        );

        unsafe { xaynai_drop(xaynai) };
        unsafe { Box::from_raw(error) };
    }

    #[test]
    fn test_xaynai_document_snippet_null() {
        let (vocab, model, docs, _, error) = setup_values();
        let (vocab, model, mut docs, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);
        let xaynai = unsafe { xaynai_new(vocab, model, error) };

        let mut invalid = vec![CDocument {
            id: docs.remove(0).id,
            snippet: unsafe { FfiStr::from_raw(null()) },
            rank: docs[0].rank,
        }];
        let invalid = invalid.as_mut_ptr();
        unsafe { xaynai_rerank(xaynai, invalid, 1, error) };
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::SnippetPointer);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
        );

        unsafe { xaynai_drop(xaynai) };
        unsafe { Box::from_raw(error) };
    }
}
