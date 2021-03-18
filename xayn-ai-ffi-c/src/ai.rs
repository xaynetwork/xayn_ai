use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::null_mut,
    slice,
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
    utils::{cstr_to_string, ErrorMsg},
};

/// Creates and initializes the Xayn AI.
///
/// # Errors
/// Returns a null pointer if:
/// - The vocab or model paths are invalid.
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
    vocab: *const u8,
    model: *const u8,
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
        let bert = bert
            .with_token_size(90)
            .expect("infallible: token size >= 2")
            .with_accents(false)
            .with_lowercase(true)
            .with_pooling(AveragePooler);
        let bert = match bert.build() {
            Ok(bert) => bert,
            Err(cause) => {
                error.set(format!("Failed to build the bert model: {}", cause));
                return null_mut();
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
/// An utf8 encoded, null-terminated message will be written to a valid error pointer.
///
/// # Safety
/// The behavior is undefined if:
/// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
/// - A non-null documents array doesn't point to an aligned, contiguous area of memory with
/// at least size [`CDocument`]s.
/// - A non-null id or snippet doesn't point to an aligned, contiguous area of memory with a
/// terminating null byte.
/// - A non-null error doesn't point to an aligned, contiguous area of memory with at least error
/// size bytes.
#[no_mangle]
pub unsafe extern "C" fn xaynai_rerank(
    xaynai: *const Reranker<Systems>,
    docs: *mut CDocument,
    size: u32,
    error: *mut u8,
    error_size: u32,
) {
    // The ai is mutated because we update the cois and prev_documents. The current code (#25) is
    // panic-safe in the sense that if a panic happen the next run of the rerank will not produce
    // invalid results (maybe an error from the feedback depending on where it panics).
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

        let size = if size == 0 {
            error.set("Failed to rerank the documents: The documents size is zero");
            return;
        } else {
            size as usize
        };
        let docs = if docs.is_null() {
            error.set("Failed to rerank the documents: The documents pointer is null");
            return;
        } else {
            unsafe { slice::from_raw_parts_mut(docs, size) }
        };
        let documents = docs.iter()
            .map(|document| {
                let id = if let Some(id) = cstr_to_string(document.id) {
                    DocumentId(id)
                } else {
                    error.set(
                        "Failed to rerank the documents: A document id is not a valid C-string pointer",
                    );
                    return None;
                };
                let snippet = if let Some(snippet) = cstr_to_string(document.snippet) {
                    snippet
                } else {
                    error.set(
                        "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
                    );
                    return None;
                };
                let rank = document.rank as usize;

                Some(Document { id, snippet, rank })
            })
            .collect::<Option<Vec<Document>>>();
        if let Some(documents) = documents {
            // TODO: use the actual reranker once it is available
            let reranks = documents
                .iter()
                .map(|document| document.rank)
                .rev()
                .collect::<DocumentsRank>();
            for (doc, rank) in docs.iter_mut().zip(reranks) {
                doc.rank = rank as u32;
            }
            error.set("");
        }
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
        ptr::null,
    };

    use super::*;
    use crate::tests::{MODEL, VOCAB};

    /// Creates values for testing.
    #[allow(clippy::type_complexity)]
    fn setup_values() -> (
        CString,
        CString,
        Vec<u8>,
        u32,
        Vec<(CString, CString, u32)>,
        u32,
    ) {
        let vocab = CString::new(VOCAB).unwrap();
        let model = CString::new(MODEL).unwrap();
        let error_size = 256;
        let error = vec![0; error_size as usize];

        let size = 10;
        let docs = (0..size)
            .map(|idx| {
                let id = CString::new(format!("{}", idx)).unwrap();
                let snippet = CString::new(format!("snippet {}", idx)).unwrap();
                let rank = idx;
                (id, snippet, rank)
            })
            .collect::<Vec<_>>();

        (vocab, model, error, error_size, docs, size)
    }

    /// Creates pointers for testing.
    fn setup_pointers<'a>(
        vocab: &CStr,
        model: &CStr,
        error: &'a mut [u8],
        docs: &[(CString, CString, u32)],
    ) -> (*const u8, *const u8, *mut u8, ErrorMsg<'a>, Vec<CDocument>) {
        let vocab = vocab.as_ptr() as *const u8;
        let model = model.as_ptr() as *const u8;
        let error_ptr = error.as_mut_ptr();
        let error_msg = error.into();
        let docs = docs
            .iter()
            .map(|(id, snippet, rank)| CDocument {
                id: id.as_ptr() as *const u8,
                snippet: snippet.as_ptr() as *const u8,
                rank: *rank,
            })
            .collect();

        (vocab, model, error_ptr, error_msg, docs)
    }

    #[test]
    fn test_xaynai_rerank() {
        let (vocab, model, mut error, error_size, docs, size) = setup_values();
        let (vocab, model, error, mut error_msg, mut docs) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            docs.as_slice(),
        );

        // new
        let xaynai = unsafe { xaynai_new(vocab, model, error, error_size) };
        assert!(!xaynai.is_null());
        assert_eq!(error_msg.to_string(), "");

        // rerank
        let reranks = docs.iter().map(|doc| doc.rank).rev().collect::<Vec<_>>();
        unsafe { xaynai_rerank(xaynai, docs.as_mut_ptr(), size, error, error_size) };
        assert!(docs.iter().map(|doc| doc.rank).eq(reranks));
        assert_eq!(error_msg.to_string(), "");

        // drop
        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_xaynai_invalid_tokenizer_paths() {
        let (vocab, model, mut error, error_size, docs, _) = setup_values();
        let (vocab, model, error, mut error_msg, _) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            docs.as_slice(),
        );

        let invalid = CString::new("").unwrap();
        let invalid = invalid.as_ptr() as *const u8;
        assert!(unsafe { xaynai_new(null(), model, error, error_size) }.is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: The vocab is not a valid C-string pointer",
        );
        assert!(unsafe { xaynai_new(invalid, model, error, error_size) }.is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)",
        );
        assert!(unsafe { xaynai_new(vocab, null(), error, error_size) }.is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: The model is not a valid C-string pointer",
        );
        assert!(unsafe { xaynai_new(vocab, invalid, error, error_size) }.is_null());
        assert_eq!(
            error_msg.to_string(),
            "Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)",
        );
    }

    #[test]
    fn test_xaynai_invalid_documents() {
        let (vocab, model, mut error, error_size, docs, size) = setup_values();
        let (vocab, model, error, mut error_msg, docs) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            docs.as_slice(),
        );
        let xaynai = unsafe { xaynai_new(vocab, model, error, error_size) };
        assert!(!xaynai.is_null());

        unsafe { xaynai_rerank(xaynai, null_mut(), size, error, error_size) };
        assert_eq!(
            error_msg.to_string(),
            "Failed to rerank the documents: The documents pointer is null",
        );

        let mut invalid = vec![CDocument {
            id: null(),
            snippet: docs[0].snippet,
            rank: docs[0].rank,
        }];
        let invalid = invalid.as_mut_ptr();
        unsafe { xaynai_rerank(xaynai, invalid, 1, error, error_size) };
        assert_eq!(
            error_msg.to_string(),
            "Failed to rerank the documents: A document id is not a valid C-string pointer",
        );

        let mut invalid = vec![CDocument {
            id: docs[0].id,
            snippet: null(),
            rank: docs[0].rank,
        }];
        let invalid = invalid.as_mut_ptr();
        unsafe { xaynai_rerank(xaynai, invalid, 1, error, error_size) };
        assert_eq!(
            error_msg.to_string(),
            "Failed to rerank the documents: A document snippet is not a valid C-string pointer",
        );

        unsafe { xaynai_drop(xaynai) };
    }

    #[test]
    fn test_xaynai_invalid_documents_size() {
        let (vocab, model, mut error, error_size, docs, _) = setup_values();
        let (vocab, model, error, mut error_msg, mut docs) = setup_pointers(
            vocab.as_c_str(),
            model.as_c_str(),
            error.as_mut_slice(),
            docs.as_slice(),
        );
        let xaynai = unsafe { xaynai_new(vocab, model, error, error_size) };
        assert!(!xaynai.is_null());

        let invalid = 0;
        unsafe { xaynai_rerank(xaynai, docs.as_mut_ptr(), invalid, error, error_size) };
        assert_eq!(
            error_msg.to_string(),
            "Failed to rerank the documents: The documents size is zero",
        );

        unsafe { xaynai_drop(xaynai) };
    }
}
