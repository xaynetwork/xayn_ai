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
    Document,
    DocumentId,
    DocumentsRank,
    Reranker,
};

use crate::c::{
    systems::{DummyAnalytics, DummyContext, DummyDatabase, DummyMab, Systems},
    utils::{cstr_to_string, ErrorMsg},
};

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
        let systems = Systems {
            database: DummyDatabase,
            bert,
            coi,
            ltr,
            context: DummyContext,
            mab: DummyMab,
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

#[no_mangle]
pub unsafe extern "C" fn xaynai_rerank(
    xaynai: *const Reranker<Systems>,
    size: u32,
    ids: *const *const u8,
    snippets: *const *const u8,
    ranks: *mut u32,
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

        let size = if size == 0 {
            error.set("Failed to rerank the documents: The size is zero");
            return;
        } else {
            size as usize
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
/// The method must be called only once on a pointer to avoid undefined behavior on double-frees.
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
