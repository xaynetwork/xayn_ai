use std::slice;

use ffi_support::{
    abort_on_panic::{call_with_result, with_abort_on_panic},
    destroy_c_string,
    implement_into_ffi_by_pointer,
    ErrorCode,
    ExternError,
};
use rubert::{AveragePooler, Builder as BertBuilder};
use xayn_ai::{
    BetaSampler,
    CoiConfiguration,
    CoiSystem,
    ConstLtr,
    Context,
    Document,
    DocumentId,
    DocumentsRank,
    MabRanking,
    Reranker,
};

use crate::{
    systems::{DummyAnalytics, DummyDatabase, Systems},
    utils::cstr_to_string,
};

/// The Xayn AI.
///
/// # Examples
/// - Create a Xayn AI with [`xaynai_new()`].
/// - Rerank documents with [`xaynai_rerank()`].
/// - Free memory with [`xaynai_drop()`] and [`error_message_drop()`].
pub struct CXaynAi(Reranker<Systems>);

implement_into_ffi_by_pointer! { CXaynAi }

/// The Xayn AI error codes.
#[repr(i32)]
#[cfg_attr(test, derive(Clone, Copy, Debug))]
#[cfg_attr(not(test), allow(dead_code))]
pub enum CXaynAiError {
    /// An irrecoverable error.
    Panic = -1,
    /// No error.
    Success = 0,
    /// A vocab null pointer error.
    VocabPointer = 1,
    /// A model null pointer error.
    ModelPointer = 2,
    /// A vocab or model file IO error.
    ReadFile = 3,
    /// A Bert builder error.
    BuildBert = 4,
    /// A Reranker builder error.
    BuildReranker = 5,
    /// A Xayn AI null pointer error.
    XaynAiPointer = 6,
    /// A documents zero size error.
    DocumentsSize = 7,
    /// A documents null pointer error.
    DocumentsPointer = 8,
    /// A document id null pointer error.
    IdPointer = 9,
    /// A document snippet null pointer error.
    SnippetPointer = 10,
}

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
    vocab: *const u8,
    model: *const u8,
    error: *mut ExternError,
) -> *mut CXaynAi {
    unsafe fn call(vocab: *const u8, model: *const u8, error: &mut ExternError) -> *mut CXaynAi {
        call_with_result(error, || {
            // bert
            let vocab = if let Some(vocab) = unsafe { cstr_to_string(vocab) } {
                vocab
            } else {
                return Err(ExternError::new_error(
                    ErrorCode::new(CXaynAiError::VocabPointer as i32),
                    "Failed to build the bert model: The vocab is not a valid C-string pointer",
                ));
            };
            let model = if let Some(model) = unsafe { cstr_to_string(model) } {
                model
            } else {
                return Err(ExternError::new_error(
                    ErrorCode::new(CXaynAiError::ModelPointer as i32),
                    "Failed to build the bert model: The model is not a valid C-string pointer",
                ));
            };
            let bert = match BertBuilder::from_files(vocab, model) {
                Ok(bert) => bert,
                Err(cause) => {
                    return Err(ExternError::new_error(
                        ErrorCode::new(CXaynAiError::ReadFile as i32),
                        format!("Failed to build the bert model: {}", cause),
                    ));
                }
            };
            let bert = bert
                .with_token_size(90)
                .expect("infallible: token size >= 2")
                .with_accents(false)
                .with_lowercase(true)
                .with_pooling(AveragePooler);
            let bert = match bert.build() {
                Ok(bert) => bert,
                Err(cause) => {
                    return Err(ExternError::new_error(
                        ErrorCode::new(CXaynAiError::BuildBert as i32),
                        format!("Failed to build the bert model: {}", cause),
                    ));
                }
            };

            // coi
            let coi = CoiSystem::new(CoiConfiguration::default());

            // ltr
            let ltr = ConstLtr(0.5);

            // context
            let context = Context;

            // mab
            let mab = MabRanking::new(BetaSampler);

            // reranker
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

    if let Some(error) = error.as_mut() {
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
pub struct CDocument {
    /// The raw pointer to the document id.
    pub id: *const u8,
    /// The raw pointer to the document snippet.
    pub snippet: *const u8,
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
/// - The documents size is zero.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null documents array doesn't point to an aligned, contiguous area of memory with
/// at least size [`CDocument`]s.
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
            // get xayn ai
            let _xaynai = if let Some(xaynai) = unsafe { xaynai.as_ref() } {
                &xaynai.0
            } else {
                return Err(ExternError::new_error(
                    ErrorCode::new(CXaynAiError::XaynAiPointer as i32),
                    "Failed to rerank the documents: The xaynai pointer is null",
                ));
            };

            // get documents
            let size = if size == 0 {
                return Err(ExternError::new_error(
                    ErrorCode::new(CXaynAiError::DocumentsSize as i32),
                    "Failed to rerank the documents: The documents size is zero",
                ));
            } else {
                size as usize
            };
            let docs = if docs.is_null() {
                return Err(ExternError::new_error(
                    ErrorCode::new(CXaynAiError::DocumentsPointer as i32),
                    "Failed to rerank the documents: The documents pointer is null",
                ));
            } else {
                unsafe { slice::from_raw_parts_mut(docs, size) }
            };
            let documents = docs.iter()
                .map(|document| {
                    let id = if let Some(id) = cstr_to_string(document.id) {
                        DocumentId(id)
                    } else {
                        return Err(ExternError::new_error(ErrorCode::new(CXaynAiError::IdPointer as i32), "Failed to rerank the documents: A document id is not a valid C-string pointer"));
                    };
                    let snippet = if let Some(snippet) = cstr_to_string(document.snippet) {
                        snippet
                    } else {
                        return Err(ExternError::new_error(ErrorCode::new(CXaynAiError::SnippetPointer as i32), "Failed to rerank the documents: A document snippet is not a valid C-string pointer"));
                    };
                    let rank = document.rank as usize;

                    Ok(Document { id, snippet, rank })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // rerank documents
            // TODO: use the actual reranker once it is available
            let reranks = documents
                .iter()
                .map(|document| document.rank)
                .rev()
                .collect::<DocumentsRank>();
            for (doc, rank) in docs.iter_mut().zip(reranks) {
                doc.rank = rank as u32;
            }

            Ok(())
        })
    }

    if let Some(error) = error.as_mut() {
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

/// Frees the memory of the error message.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null error message doesn't point to memory allocated by [`xaynai_new()`] or
/// [`xaynai_rerank()`].
/// - A non-null error message is freed more than once.
/// - A non-null error message is accessed after being freed.
#[no_mangle]
pub unsafe extern "C" fn error_message_drop(error: *mut ExternError) {
    with_abort_on_panic(|| {
        if let Some(error) = error.as_mut() {
            unsafe { destroy_c_string(error.get_raw_message() as *mut _) }
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
    fn setup_pointers(
        vocab: &CStr,
        model: &CStr,
        docs: &[(CString, CString, u32)],
        error: ExternError,
    ) -> (*const u8, *const u8, Vec<CDocument>, *mut ExternError) {
        let vocab = vocab.as_ptr() as *const u8;
        let model = model.as_ptr() as *const u8;

        let docs = docs
            .iter()
            .map(|(id, snippet, rank)| CDocument {
                id: id.as_ptr() as *const u8,
                snippet: snippet.as_ptr() as *const u8,
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
    fn test_xaynai_invalid_tokenizer_paths() {
        let (vocab, model, docs, _, error) = setup_values();
        let (vocab, model, _, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);

        let invalid = CString::new("").unwrap();
        let invalid = invalid.as_ptr() as *const u8;
        assert!(unsafe { xaynai_new(null(), model, error) }.is_null());
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::VocabPointer);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to build the bert model: The vocab is not a valid C-string pointer",
        );

        assert!(unsafe { xaynai_new(invalid, model, error) }.is_null());
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::ReadFile);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)",
        );

        assert!(unsafe { xaynai_new(vocab, null(), error) }.is_null());
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::ModelPointer);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to build the bert model: The model is not a valid C-string pointer",
        );

        assert!(unsafe { xaynai_new(vocab, invalid, error) }.is_null());
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::ReadFile);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)",
        );

        unsafe { Box::from_raw(error) };
    }

    #[test]
    fn test_xaynai_invalid_documents_size() {
        let (vocab, model, docs, _, error) = setup_values();
        let (vocab, model, mut docs, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);
        let xaynai = unsafe { xaynai_new(vocab, model, error) };

        let invalid = 0;
        unsafe { xaynai_rerank(xaynai, docs.as_mut_ptr(), invalid, error) };
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::DocumentsSize);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to rerank the documents: The documents size is zero",
        );

        unsafe { xaynai_drop(xaynai) };
        unsafe { Box::from_raw(error) };
    }

    #[test]
    fn test_xaynai_invalid_documents() {
        let (vocab, model, docs, size, error) = setup_values();
        let (vocab, model, docs, error) =
            setup_pointers(vocab.as_c_str(), model.as_c_str(), docs.as_slice(), error);
        let xaynai = unsafe { xaynai_new(vocab, model, error) };

        unsafe { xaynai_rerank(xaynai, null_mut(), size, error) };
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::DocumentsPointer);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to rerank the documents: The documents pointer is null",
        );

        let mut invalid = vec![CDocument {
            id: null(),
            snippet: docs[0].snippet,
            rank: docs[0].rank,
        }];
        let invalid = invalid.as_mut_ptr();
        unsafe { xaynai_rerank(xaynai, invalid, 1, error) };
        assert_eq!(unsafe { error_code(error) }, CXaynAiError::IdPointer);
        assert_eq!(
            unsafe { error_message(error) },
            "Failed to rerank the documents: A document id is not a valid C-string pointer",
        );

        let mut invalid = vec![CDocument {
            id: docs[0].id,
            snippet: null(),
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
