use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::null_mut,
    slice,
};

use itertools::izip;
use rubert::{Builder as BertBuilder, Pooler};
use xayn_ai::{
    CoiConfiguration,
    CoiSystem,
    ConstLtr,
    Context,
    Document,
    DocumentId,
    DocumentsRank,
    Reranker,
};

use crate::c::{
    systems::{DummyAnalytics, DummyDatabase, DummyMab, Systems},
    utils::{cstr_to_string, ErrorMsg},
};

/// Creates and initializes the Xayn AI.
///
/// # Errors
/// Aborts and returns a null pointer if:
/// - The vocab or model paths are invalid.
/// - The batch or token sizes are invalid.
///
/// An utf8 encoded, null-terminated message will be written to a valid error pointer.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null vocab or model path doesn't point to an aligned, contiguous area of memory with a
/// terminating null byte.
/// - A non-null error doesn't point to an aligned, contiguous area of memory with at least error
/// size bytes.
#[no_mangle]
pub unsafe extern "C" fn xaynai_new(
    // bert
    vocab: *const u8,
    model: *const u8,
    batch_size: u32,
    token_size: u32,
    // coi
    shift_factor: f32,
    threshold: f32,
    // error
    error: *mut u8,
    error_size: u32,
) -> *mut Reranker<Systems> {
    catch_unwind(move || {
        let mut error = unsafe { ErrorMsg::new(error, error_size) };

        // bert
        let vocab = if let Some(vocab) = unsafe { cstr_to_string(vocab) } {
            vocab
        } else {
            error.set("Failed to build the bert model: The vocab is not a valid C-string pointer");
            return null_mut();
        };
        let model = if let Some(model) = unsafe { cstr_to_string(model) } {
            model
        } else {
            error.set("Failed to build the bert model: The model is not a valid C-string pointer");
            return null_mut();
        };
        let bert = match BertBuilder::from_files(vocab, model) {
            Ok(bert) => bert,
            Err(cause) => {
                error.set(format!("Failed to build the bert model: {}", cause));
                return null_mut();
            }
        };
        let bert = match bert.with_batch_size(batch_size as usize) {
            Ok(bert) => bert,
            Err(cause) => {
                error.set(format!("Failed to build the bert model: {}", cause));
                return null_mut();
            }
        };
        let bert = match bert.with_token_size(token_size as usize) {
            Ok(bert) => bert,
            Err(cause) => {
                error.set(format!("Failed to build the bert model: {}", cause));
                return null_mut();
            }
        };
        // accents, lowercase and pooler have been fixed before, we should make them configurable
        // at one point, but that will be a breaking change for the embeddings used by the ai
        let bert = bert
            .with_strip_accents(true)
            .with_lowercase(true)
            .with_pooling(Pooler::Average);
        let bert = match bert.build() {
            Ok(bert) => bert,
            Err(cause) => {
                error.set(format!("Failed to build the bert model: {}", cause));
                return null_mut();
            }
        };

        // coi
        let coi = CoiConfiguration {
            shift_factor,
            threshold,
        };
        let coi = CoiSystem::new(coi);

        // ltr
        let ltr = ConstLtr(0.5);

        // reranker
        // TODO: use the reranker builder once it is available
        let systems = Systems {
            // TODO: use the actual database once it is available
            database: DummyDatabase,
            bert,
            coi,
            ltr,
            context: Context,
            // TODO: use the actual mab once it is available
            mab: DummyMab,
            // TODO: use the actual analytics once it is available
            analytics: DummyAnalytics,
        };
        let reranker = match Reranker::new(systems) {
            Ok(reranker) => reranker,
            Err(cause) => {
                error.set(format!("Failed to build the reranker: {}", cause));
                return null_mut();
            }
        };

        error.set("");
        Box::into_raw(Box::new(reranker))
    })
    .unwrap_or_else(|cause| {
        unsafe { ErrorMsg::new(error, error_size) }.set(format!(
            "Panicked while creating the reranker: {:?}.",
            cause
        ));
        null_mut()
    })
}

/// Reranks the documents with the Xayn AI.
///
/// Each document is represented as an id, a snippet and a rank. The reranked order is written to
/// the ranks array.
///
/// # Errors
/// Aborts without changing the ranks if:
/// - The xaynai is null.
/// - The ids, snippets or ranks are invalid.
/// - The document size is zero.
///
/// An utf8 encoded, null-terminated message will be written to a valid error pointer.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null ids or snippets array doesn't point to an aligned, contiguous area of memory with
/// at least doc size pointers.
/// - A non-null ranks array doesn't point to an aligned, contiguous area of memory with at least
/// doc size integers.
/// - A non-null id or snippet doesn't point to an aligned, contiguous area of memory with a
/// terminating null byte.
/// - A non-null error doesn't point to an aligned, contiguous area of memory with at least error
/// size bytes.
#[no_mangle]
pub unsafe extern "C" fn xaynai_rerank(
    xaynai: *const Reranker<Systems>,
    ids: *const *const u8,
    snippets: *const *const u8,
    ranks: *mut u32,
    doc_size: u32,
    error: *mut u8,
    error_size: u32,
) {
    // TODO: check if the ai gets mutated during reranking, if so drop the ai in case of a panic
    let xaynai = AssertUnwindSafe(xaynai);

    catch_unwind(move || {
        let mut error = unsafe { ErrorMsg::new(error, error_size) };
        let error = &mut error;

        let _xaynai = if let Some(xaynai) = unsafe { xaynai.as_ref() } {
            xaynai
        } else {
            error.set("Failed to rerank the documents: The xaynai pointer is null");
            return;
        };

        let size = if doc_size == 0 {
            error.set("Failed to rerank the documents: The document size is zero");
            return;
        } else {
            doc_size as usize
        };
        let ids = if ids.is_null() {
            error.set("Failed to rerank the documents: The ids pointer is null");
            return;
        } else {
            unsafe { slice::from_raw_parts(ids, size) }
        };
        let snippets = if snippets.is_null() {
            error.set("Failed to rerank the documents: The snippets pointer is null");
            return;
        } else {
            unsafe { slice::from_raw_parts(snippets, size) }
        };
        let ranks = if ranks.is_null() {
            error.set("Failed to rerank the documents: The ranks pointer is null");
            return;
        } else {
            unsafe { slice::from_raw_parts_mut(ranks, size) }
        };

        let documents = if let Some(documents) = izip!(ids, snippets, ranks.iter())
            .map(|(id, snippet, rank)| {
                let id = if let Some(id) = cstr_to_string(*id) {
                    DocumentId(id)
                } else {
                    error.set(
                        "Failed to rerank the documents: An id is not a valid C-string pointer",
                    );
                    return None;
                };
                let snippet = if let Some(snippet) = cstr_to_string(*snippet) {
                    snippet
                } else {
                    error.set(
                        "Failed to rerank the documents: A snippet is not a valid C-string pointer",
                    );
                    return None;
                };
                let rank = *rank as usize;

                Some(Document { id, snippet, rank })
            })
            .collect::<Option<Vec<Document>>>()
        {
            documents
        } else {
            return;
        };

        // TODO: use the actual reranker once it is available
        let reranks = documents
            .iter()
            .map(|document| document.rank)
            .rev()
            .collect::<DocumentsRank>();

        for (rank, rerank) in izip!(ranks, reranks) {
            *rank = rerank as u32;
        }
        error.set("");
    })
    .unwrap_or_else(|cause| {
        unsafe { ErrorMsg::new(error, error_size) }.set(format!(
            "Panicked while reranking the documents: {:?}.",
            cause
        ));
    })
}

/// Frees the memory of the Xayn AI.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null xaynai is freed more than once.
#[no_mangle]
pub unsafe extern "C" fn xaynai_drop(xaynai: *mut Reranker<Systems>) {
    // the reranker gets dropped anyways
    let xaynai = AssertUnwindSafe(xaynai);

    catch_unwind(|| {
        if !xaynai.is_null() {
            unsafe { Box::from_raw(xaynai.0) };
        }
    })
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::{CStr, CString},
        iter,
        ptr::null,
    };

    use super::*;

    /// Creates values for testing.
    fn setup_vals() -> (
        CString,
        CString,
        u32,
        u32,
        f32,
        f32,
        Vec<u8>,
        u32,
        u32,
        Vec<CString>,
        Vec<CString>,
        Vec<u32>,
    ) {
        let vocab = CString::new("../data/rubert_v0000/vocab.txt").unwrap();
        let model = CString::new("../data/rubert_v0000/model.onnx").unwrap();
        let batch_size = 10;
        let token_size = 64;
        let shift_factor = 0.1;
        let threshold = 10.0;
        let error_size = 256;
        let error = vec![0; error_size as usize];

        let doc_size = 10;
        let ids = (0..doc_size)
            .map(|id| CString::new(format!("{}", id)).unwrap())
            .collect();
        let snippets = (0..doc_size)
            .map(|id| CString::new(format!("snippet {}", id)).unwrap())
            .collect();
        let ranks = (0..doc_size).collect();

        (
            vocab,
            model,
            batch_size,
            token_size,
            shift_factor,
            threshold,
            error,
            error_size,
            doc_size,
            ids,
            snippets,
            ranks,
        )
    }

    /// Creates pointers for testing.
    fn setup_refs<'a>(
        vocab: &CStr,
        model: &CStr,
        error: &'a mut [u8],
        ids: &[CString],
        snippets: &[CString],
        ranks: &mut [u32],
    ) -> (
        *const u8,
        *const u8,
        *mut u8,
        ErrorMsg<'a>,
        Vec<*const u8>,
        Vec<*const u8>,
        *mut u32,
    ) {
        let vocab = vocab.as_ptr() as *const u8;
        let model = model.as_ptr() as *const u8;
        let error_ptr = error.as_mut_ptr();
        let error_msg = error.into();

        let ids = ids.iter().map(|id| id.as_ptr() as *const u8).collect();
        let snippets = snippets
            .iter()
            .map(|snippet| snippet.as_ptr() as *const u8)
            .collect();
        let ranks = ranks.as_mut_ptr();

        (vocab, model, error_ptr, error_msg, ids, snippets, ranks)
    }

    /// Creates pointers of pointers for testing.
    fn setup_refs_refs(
        ids: &[*const u8],
        snippets: &[*const u8],
    ) -> (*const *const u8, *const *const u8) {
        let ids = ids.as_ptr();
        let snippets = snippets.as_ptr();

        (ids, snippets)
    }

    #[test]
    fn test_xaynai_rerank() {
        let (
            vocab,
            model,
            batch_size,
            token_size,
            shift_factor,
            threshold,
            mut error,
            error_size,
            doc_size,
            ids,
            snippets,
            mut ranks_val,
        ) = setup_vals();
        let reranks_val = ranks_val.iter().copied().rev().collect::<Vec<_>>();
        let (vocab, model, error, mut error_msg, ids, snippets, ranks) = setup_refs(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            ids.as_slice(),
            snippets.as_slice(),
            ranks_val.as_mut_slice(),
        );
        let (ids, snippets) = setup_refs_refs(ids.as_slice(), snippets.as_slice());

        // new
        let xaynai = unsafe {
            xaynai_new(
                vocab,
                model,
                batch_size,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        };
        assert!(!xaynai.is_null());
        assert_eq!(error_msg.to_string(), "");

        // rerank
        unsafe { xaynai_rerank(xaynai, ids, snippets, ranks, doc_size, error, error_size) };
        assert_eq!(ranks_val, reranks_val);
        assert_eq!(error_msg.to_string(), "");

        // drop
        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_xaynai_invalid_tokenizer_paths() {
        let (
            vocab,
            model,
            batch_size,
            token_size,
            shift_factor,
            threshold,
            mut error,
            error_size,
            _,
            ids,
            snippets,
            mut ranks_val,
        ) = setup_vals();
        let (vocab, model, error, mut error_msg, _, _, _) = setup_refs(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            ids.as_slice(),
            snippets.as_slice(),
            ranks_val.as_mut_slice(),
        );

        let null_ = null();
        let invalid = CString::new("").unwrap();
        let invalid = invalid.as_ptr() as *const u8;
        assert!(unsafe {
            xaynai_new(
                null_,
                model,
                batch_size,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        }
        .is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: The vocab is not a valid C-string pointer",
        );
        assert!(unsafe {
            xaynai_new(
                invalid,
                model,
                batch_size,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        }
        .is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2).",
        );
        assert!(unsafe {
            xaynai_new(
                vocab,
                null_,
                batch_size,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        }
        .is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: The model is not a valid C-string pointer",
        );
        assert!(unsafe {
            xaynai_new(
                vocab,
                invalid,
                batch_size,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        }
        .is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2).",
        );
    }

    #[test]
    fn test_xaynai_invalid_tokenizer_sizes() {
        let (
            vocab,
            model,
            batch_size,
            token_size,
            shift_factor,
            threshold,
            mut error,
            error_size,
            _,
            ids,
            snippets,
            mut ranks_val,
        ) = setup_vals();
        let (vocab, model, error, mut error_msg, _, _, _) = setup_refs(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            ids.as_slice(),
            snippets.as_slice(),
            ranks_val.as_mut_slice(),
        );

        let invalid = 0;
        assert!(unsafe {
            xaynai_new(
                vocab,
                model,
                invalid,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        }
        .is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: The batch size must be greater than zero.",
        );
        assert!(unsafe {
            xaynai_new(
                vocab,
                model,
                batch_size,
                invalid,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        }
        .is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: The token size must be greater than two to allow for special tokens.",
        );
    }

    #[test]
    fn test_xaynai_invalid_document_ids() {
        let (
            vocab,
            model,
            batch_size,
            token_size,
            shift_factor,
            threshold,
            mut error,
            error_size,
            doc_size,
            ids,
            snippets,
            mut ranks_val,
        ) = setup_vals();
        let (vocab, model, error, mut error_msg, ids, snippets, ranks) = setup_refs(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            ids.as_slice(),
            snippets.as_slice(),
            ranks_val.as_mut_slice(),
        );
        let (_, snippets) = setup_refs_refs(ids.as_slice(), snippets.as_slice());
        let xaynai = unsafe {
            xaynai_new(
                vocab,
                model,
                batch_size,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        };
        assert!(!xaynai.is_null());

        let null_ = null();
        unsafe { xaynai_rerank(xaynai, null_, snippets, ranks, doc_size, error, error_size) };
        assert_eq!(
            error_msg.to_string(),
            "Failed to rerank the documents: The ids pointer is null",
        );
        let null_ = null();
        for idx in 0..doc_size as usize {
            let invalid = ids
                .iter()
                .take(idx)
                .copied()
                .chain(iter::once(null_))
                .chain(ids.iter().skip(idx).copied())
                .collect::<Vec<*const u8>>();
            let invalid = invalid.as_ptr();
            unsafe {
                xaynai_rerank(
                    xaynai, invalid, snippets, ranks, doc_size, error, error_size,
                )
            };
            assert_eq!(
                error_msg.to_string(),
                "Failed to rerank the documents: An id is not a valid C-string pointer",
            );
        }

        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_xaynai_invalid_document_snippets() {
        let (
            vocab,
            model,
            batch_size,
            token_size,
            shift_factor,
            threshold,
            mut error,
            error_size,
            doc_size,
            ids,
            snippets,
            mut ranks_val,
        ) = setup_vals();
        let (vocab, model, error, mut error_msg, ids, snippets, ranks) = setup_refs(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            ids.as_slice(),
            snippets.as_slice(),
            ranks_val.as_mut_slice(),
        );
        let (ids, _) = setup_refs_refs(ids.as_slice(), snippets.as_slice());
        let xaynai = unsafe {
            xaynai_new(
                vocab,
                model,
                batch_size,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        };
        assert!(!xaynai.is_null());

        let null_ = null();
        unsafe { xaynai_rerank(xaynai, ids, null_, ranks, doc_size, error, error_size) };
        assert_eq!(
            error_msg.to_string(),
            "Failed to rerank the documents: The snippets pointer is null",
        );
        let null_ = null();
        for idx in 0..doc_size as usize {
            let invalid = snippets
                .iter()
                .take(idx)
                .copied()
                .chain(iter::once(null_))
                .chain(snippets.iter().skip(idx).copied())
                .collect::<Vec<*const u8>>();
            let invalid = invalid.as_ptr();
            unsafe { xaynai_rerank(xaynai, ids, invalid, ranks, doc_size, error, error_size) };
            assert_eq!(
                error_msg.to_string(),
                "Failed to rerank the documents: A snippet is not a valid C-string pointer",
            );
        }

        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_xaynai_invalid_document_ranks() {
        let (
            vocab,
            model,
            batch_size,
            token_size,
            shift_factor,
            threshold,
            mut error,
            error_size,
            doc_size,
            ids,
            snippets,
            mut ranks_val,
        ) = setup_vals();
        let (vocab, model, error, mut error_msg, ids, snippets, _) = setup_refs(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            ids.as_slice(),
            snippets.as_slice(),
            ranks_val.as_mut_slice(),
        );
        let (ids, snippets) = setup_refs_refs(ids.as_slice(), snippets.as_slice());
        let xaynai = unsafe {
            xaynai_new(
                vocab,
                model,
                batch_size,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        };
        assert!(!xaynai.is_null());

        let null_ = null_mut();
        unsafe { xaynai_rerank(xaynai, ids, snippets, null_, doc_size, error, error_size) };
        assert_eq!(
            error_msg.to_string(),
            "Failed to rerank the documents: The ranks pointer is null",
        );

        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_xaynai_invalid_document_size() {
        let (
            vocab,
            model,
            batch_size,
            token_size,
            shift_factor,
            threshold,
            mut error,
            error_size,
            _,
            ids,
            snippets,
            mut ranks_val,
        ) = setup_vals();
        let (vocab, model, error, mut error_msg, ids, snippets, ranks) = setup_refs(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            ids.as_slice(),
            snippets.as_slice(),
            ranks_val.as_mut_slice(),
        );
        let (ids, snippets) = setup_refs_refs(ids.as_slice(), snippets.as_slice());
        let xaynai = unsafe {
            xaynai_new(
                vocab,
                model,
                batch_size,
                token_size,
                shift_factor,
                threshold,
                error,
                error_size,
            )
        };
        assert!(!xaynai.is_null());

        let invalid = 0;
        unsafe { xaynai_rerank(xaynai, ids, snippets, ranks, invalid, error, error_size) };
        assert_eq!(
            error_msg.to_string(),
            "Failed to rerank the documents: The document size is zero",
        );

        unsafe { xaynai_drop(xaynai) };
    }
}
