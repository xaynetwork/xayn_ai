use js_sys::Uint8Array;
use wasm_bindgen::prelude::{wasm_bindgen, JsValue};
use xayn_ai::{Builder, Document, DocumentHistory, Reranker};

use super::{error::CCode, utils::IntoJsResult};

#[wasm_bindgen]
/// The Xayn AI.
pub struct WXaynAi(Reranker);

#[wasm_bindgen]
impl WXaynAi {
    #[wasm_bindgen(constructor)]
    /// Creates and initializes the Xayn AI.
    ///
    /// Requires the vocabulary and model of the tokenizer/embedder. Optionally accepts the serialized
    /// reranker database, otherwise creates a new one.
    ///
    /// # Errors
    ///
    /// - The `vocab` or `model` data are invalid.
    /// - The the data of the `serialized` database is invalid.
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
        JsValue::from_serde(&self.0.analytics()).unwrap()
    }
}

#[cfg(target_arch = "wasm32")]
#[cfg(test)]
mod tests {
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    /// Path to the current vocabulary file.
    pub const VOCAB: &[u8] = include_bytes!("../../data/rubert_v0000/vocab.txt");

    /// Path to the current onnx model file.
    pub const MODEL: &[u8] = include_bytes!("../../data/rubert_v0000/model.onnx");

    #[wasm_bindgen_test]
    fn test_reranker() {
        let mut xaynai = WXaynAi::new(VOCAB, MODEL, None).unwrap();

        let document = Document {
            id: "1".into(),
            rank: 0,
            snippet: "abc".to_string(),
        };
        let js_document = JsValue::from_serde(&document).unwrap();

        let ranks = xaynai.rerank(vec![], vec![js_document]).unwrap();
        assert_eq!(ranks, [0]);

        let ser = xaynai.serialize().unwrap();
        assert!(ser.length() != 0)
    }
}
