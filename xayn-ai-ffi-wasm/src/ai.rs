use js_sys::Uint8Array;
use wasm_bindgen::prelude::{wasm_bindgen, JsValue};
use xayn_ai::{Builder, Document, DocumentHistory, Reranker};

use super::{error::CCode, utils::IntoJsResult};

/// The Xayn AI.
#[wasm_bindgen]
pub struct WXaynAi(Reranker);

#[wasm_bindgen]
impl WXaynAi {
    /// Creates and initializes the Xayn AI.
    ///
    /// Requires the vocabulary and model of the tokenizer/embedder. Optionally accepts the serialized
    /// reranker database, otherwise creates a new one.
    ///
    /// # Errors
    ///
    /// - The `vocab` or `model` data are invalid.
    /// - The the data of the `serialized` database is invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(
        vocab: &[u8],
        model: &[u8],
        serialized: Option<Box<[u8]>>,
    ) -> Result<WXaynAi, JsValue> {
        console_error_panic_hook::set_once();

        Builder::default()
            .with_serialized_database(&serialized.unwrap_or_default())
            .map_err(|cause| {
                CCode::RerankerDeserialization.with_context(format!(
                    "Failed to deserialize the reranker database: {}",
                    cause
                ))
            })
            .into_js_result()?
            .with_bert_from_reader(vocab, model)
            .build()
            .map(WXaynAi)
            .map_err(|cause| {
                CCode::InitAi.with_context(format!("Failed to initialize the ai: {}", cause))
            })
            .into_js_result()
    }

    /// Reranks the documents with the Xayn AI.
    ///
    /// # Errors
    ///
    /// - The deserialization of a document `history` fails.
    /// - The deserialization of a `document` fails.
    pub fn rerank(
        &mut self,
        history: Vec<JsValue>,
        documents: Vec<JsValue>,
    ) -> Result<Vec<usize>, JsValue> {
        let history = history
            .iter()
            .map(JsValue::into_serde)
            .collect::<Result<Vec<DocumentHistory>, _>>()
            .map_err(|cause| {
                CCode::HistoriesDeserialization.with_context(format!(
                    "Failed to deserialize the collection of histories: {}",
                    cause
                ))
            })
            .into_js_result()?;
        let documents = documents
            .iter()
            .map(JsValue::into_serde)
            .collect::<Result<Vec<Document>, _>>()
            .map_err(|cause| {
                CCode::DocumentsDeserialization.with_context(format!(
                    "Failed to deserialize the collection of documents: {}",
                    cause
                ))
            })
            .into_js_result()?;
        Ok(self.0.rerank(&history, &documents))
    }

    /// Serializes the database of the reranker.
    ///
    /// # Errors
    ///
    /// - The serialization of the database fails.
    pub fn serialize(&self) -> Result<Uint8Array, JsValue> {
        self.0
            .serialize()
            .map(|bytes| bytes.as_slice().into())
            .map_err(|cause| {
                CCode::RerankerSerialization
                    .with_context(format!("Failed to serialize the reranker: {}", cause))
            })
            .into_js_result()
    }

    /// Retrieves faults which might occur during reranking.
    ///
    /// Faults can range from warnings to errors which are handled in some default way internally.
    pub fn faults(&self) -> Vec<JsValue> {
        self.0
            .errors()
            .iter()
            .map(|error| {
                JsValue::from_serde(&CCode::Fault.with_context(error.to_string())).unwrap()
            })
            .collect()
    }

    /// Retrieves the analytics which were collected in the penultimate reranking.
    pub fn analytics(&self) -> JsValue {
        JsValue::from_serde(&self.0.analytics()).expect("Failed to serialize the analytics")
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    #[cfg(all(feature = "browser", feature = "node"))]
    compile_error!("feature `browser` and `node` may not be used at the same time");

    #[cfg(feature = "browser")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    use super::*;

    use std::iter::repeat;

    use itertools::izip;
    use wasm_bindgen_test::wasm_bindgen_test;
    use xayn_ai::{Relevance, UserFeedback};

    use crate::error::ExternError;

    /// Path to the current vocabulary file.
    const VOCAB: &[u8] = include_bytes!("../../data/rubert_v0001/vocab.txt");

    /// Path to the current onnx model file.
    const MODEL: &[u8] = include_bytes!("../../data/rubert_v0001/smbert.onnx");

    impl std::fmt::Debug for WXaynAi {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            f.debug_struct("WXaynAi").finish()
        }
    }

    fn test_histories() -> Vec<JsValue> {
        let len = 6;
        let ids = (0..len).map(|idx| idx.to_string()).collect::<Vec<_>>();

        let relevances = repeat(Relevance::Low)
            .take(len / 2)
            .chain(repeat(Relevance::High).take(len - len / 2));
        let feedbacks = repeat(UserFeedback::Irrelevant)
            .take(len / 2)
            .chain(repeat(UserFeedback::Relevant).take(len - len / 2));

        let history = izip!(ids, relevances, feedbacks)
            .map(|(id, relevance, user_feedback)| {
                JsValue::from_serde(&DocumentHistory {
                    id: id.as_str().into(),
                    relevance,
                    user_feedback,
                })
                .unwrap()
            })
            .collect::<Vec<_>>();

        history
    }

    fn test_documents() -> Vec<JsValue> {
        let len = 10;
        let ids = (0..len).map(|idx| idx.to_string()).collect::<Vec<_>>();

        let snippets = (0..len)
            .map(|idx| format!("snippet {}", idx))
            .collect::<Vec<_>>();
        let ranks = 0..len as usize;

        let document = izip!(ids, snippets, ranks)
            .map(|(id, snippet, rank)| {
                JsValue::from_serde(&Document {
                    id: id.as_str().into(),
                    snippet,
                    rank,
                })
                .unwrap()
            })
            .collect::<Vec<_>>();

        document
    }

    #[wasm_bindgen_test]
    fn test_rerank() {
        let mut xaynai = WXaynAi::new(VOCAB, MODEL, None).unwrap();
        xaynai.rerank(test_histories(), test_documents()).unwrap();
    }

    #[wasm_bindgen_test]
    fn test_serialize() {
        let xaynai = WXaynAi::new(VOCAB, MODEL, None).unwrap();
        xaynai.serialize().unwrap();
    }

    #[wasm_bindgen_test]
    fn test_faults() {
        let xaynai = WXaynAi::new(VOCAB, MODEL, None).unwrap();
        let faults = xaynai.faults();
        assert!(faults.is_empty());
    }

    #[wasm_bindgen_test]
    fn test_analytics() {
        let xaynai = WXaynAi::new(VOCAB, MODEL, None).unwrap();
        let analytics = xaynai.analytics();
        assert!(analytics.is_null());
    }

    #[wasm_bindgen_test]
    fn test_vocab_invalid() {
        let error = WXaynAi::new(&[], MODEL, None)
            .unwrap_err()
            .into_serde::<ExternError>()
            .unwrap();

        assert_eq!(error.code, CCode::InitAi as i32);
        assert!(error
            .message
            .contains("Failed to initialize the ai: Failed to build the tokenizer: "));
    }

    #[wasm_bindgen_test]
    fn test_model_invalid() {
        let error = WXaynAi::new(VOCAB, &[1], None)
            .unwrap_err()
            .into_serde::<ExternError>()
            .unwrap();

        assert_eq!(error.code, CCode::InitAi as i32);
        assert!(error
            .message
            .contains("Failed to initialize the ai: Failed to build the model: "));
    }

    #[wasm_bindgen_test]
    fn test_history_invalid() {
        let mut xaynai = WXaynAi::new(VOCAB, MODEL, None).unwrap();
        let error = xaynai
            .rerank(vec![JsValue::from("invalid")], test_documents())
            .unwrap_err()
            .into_serde::<ExternError>()
            .unwrap();

        assert_eq!(error.code, CCode::HistoriesDeserialization as i32);
        assert!(error
            .message
            .contains("Failed to deserialize the collection of histories: invalid type: "));
    }

    #[wasm_bindgen_test]
    fn test_history_empty() {
        let mut xaynai = WXaynAi::new(VOCAB, MODEL, None).unwrap();
        xaynai.rerank(vec![], test_documents()).unwrap();
    }

    #[wasm_bindgen_test]
    fn test_documents_invalid() {
        let mut xaynai = WXaynAi::new(VOCAB, MODEL, None).unwrap();
        let error = xaynai
            .rerank(test_histories(), vec![JsValue::from("invalid")])
            .unwrap_err()
            .into_serde::<ExternError>()
            .unwrap();

        assert_eq!(error.code, CCode::DocumentsDeserialization as i32);
        assert!(error
            .message
            .contains("Failed to deserialize the collection of documents: invalid type: "));
    }

    #[wasm_bindgen_test]
    fn test_documents_empty() {
        let mut xaynai = WXaynAi::new(VOCAB, MODEL, None).unwrap();
        xaynai.rerank(test_histories(), vec![]).unwrap();
    }

    #[wasm_bindgen_test]
    fn test_serialized_empty() {
        WXaynAi::new(VOCAB, MODEL, Some(Box::new([]))).unwrap();
    }

    #[wasm_bindgen_test]
    fn test_serialized_invalid() {
        let error = WXaynAi::new(VOCAB, MODEL, Some(Box::new([1, 2, 3])))
            .unwrap_err()
            .into_serde::<ExternError>()
            .unwrap();

        assert_eq!(error.code, CCode::RerankerDeserialization as i32);
        assert!(error.message.contains(
            "Failed to deserialize the reranker database: Unsupported serialized data. "
        ));
    }
}
