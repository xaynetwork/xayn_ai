/// This function does nothing.
///
/// Calling this prevents Swift to drop the library.
#[no_mangle]
pub extern "C" fn dummy_function() {}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        ffi::{CStr, CString},
        ptr::null,
    };

    use ffi_support::{ExternError, FfiStr};

    use crate::{
        database::Database,
        document::{CDocument, CFeedback, CHistory, CRelevance},
        tests::{MODEL, VOCAB},
    };

    /// Creates values for testing.
    #[allow(clippy::type_complexity)]
    pub fn setup_values() -> (
        CString,
        CString,
        Vec<(CString, CRelevance, CFeedback)>,
        u32,
        Vec<(CString, CString, u32)>,
        u32,
        ExternError,
    ) {
        let vocab = CString::new(VOCAB).unwrap();
        let model = CString::new(MODEL).unwrap();

        let hist_size = 6;
        let hist = (0..hist_size / 2)
            .map(|idx| {
                let id = CString::new(idx.to_string()).unwrap();
                let relevance = CRelevance::Low;
                let feedback = CFeedback::Irrelevant;
                (id, relevance, feedback)
            })
            .chain((hist_size / 2..hist_size).map(|idx| {
                let id = CString::new(idx.to_string()).unwrap();
                let relevance = CRelevance::High;
                let feedback = CFeedback::Relevant;
                (id, relevance, feedback)
            }))
            .collect::<Vec<_>>();

        let docs_size = 10;
        let docs = (0..docs_size)
            .map(|idx| {
                let id = CString::new(idx.to_string()).unwrap();
                let snippet = CString::new(format!("snippet {}", idx)).unwrap();
                let rank = idx;
                (id, snippet, rank)
            })
            .collect::<Vec<_>>();

        let error = ExternError::default();

        (vocab, model, hist, hist_size, docs, docs_size, error)
    }

    /// Creates pointers for testing.
    pub fn setup_pointers<'a>(
        vocab: &'a CStr,
        model: &'a CStr,
        hist: &'a [(CString, CRelevance, CFeedback)],
        docs: &'a [(CString, CString, u32)],
        error: &mut ExternError,
    ) -> (
        FfiStr<'a>,
        FfiStr<'a>,
        *const Database,
        Vec<CHistory<'a>>,
        Vec<CDocument<'a>>,
        *mut ExternError,
    ) {
        let vocab = FfiStr::from_cstr(vocab);
        let model = FfiStr::from_cstr(model);

        let database = null();

        let hist = hist
            .iter()
            .map(|(id, relevance, feedback)| CHistory {
                id: FfiStr::from_cstr(id),
                relevance: *relevance,
                feedback: *feedback,
            })
            .collect();

        let docs = docs
            .iter()
            .map(|(id, snippet, rank)| CDocument {
                id: FfiStr::from_cstr(id),
                snippet: FfiStr::from_cstr(snippet),
                rank: *rank,
            })
            .collect();

        let error = error as *mut _;

        (vocab, model, database, hist, docs, error)
    }

    /// Cleans up the leaked memory of the test values.
    pub fn drop_values(
        vocab: CString,
        model: CString,
        hist: Vec<(CString, CRelevance, CFeedback)>,
        docs: Vec<(CString, CString, u32)>,
        error: ExternError,
    ) {
        vocab.into_string().unwrap();
        model.into_string().unwrap();

        for (id, _, _) in hist.into_iter() {
            id.into_string().unwrap();
        }

        for (id, snippet, _) in docs.into_iter() {
            id.into_string().unwrap();
            snippet.into_string().unwrap();
        }

        unsafe { error.manually_release() };
    }
}
