use std::collections::HashMap;

use js_sys::Uint8Array;
use wasm_bindgen::prelude::{wasm_bindgen, JsValue};
use xayn_ai::{Builder, Document, DocumentHistory, Reranker};

use super::utils::{ExternError, IntoJsResult};

#[wasm_bindgen]
pub struct WXaynAi(Reranker);

#[wasm_bindgen]
impl WXaynAi {
    #[wasm_bindgen(constructor)]
    pub fn new(
        vocab: &[u8],
        model: &[u8],
        serialized: Option<Box<[u8]>>,
    ) -> Result<WXaynAi, JsValue> {
        Builder::default()
            .with_serialized_database(&serialized.unwrap_or_default())
            .map_err(|cause| ExternError {
                code: 1,
                msg: format!("Failed to deserialize the reranker database: {}", cause),
            })
            .into_js_result()?
            .with_bert_from_reader(vocab, model)
            .build()
            .map(WXaynAi)
            .map_err(|cause| ExternError {
                code: 2,
                msg: format!("Failed to initialize the ai: {}", cause),
            })
            .into_js_result()
    }

    pub fn rerank(
        &mut self,
        // Vec<JsValue> behaves like Box<[JsValue]> here
        // https://rustwasm.github.io/wasm-bindgen/api/wasm_bindgen/convert/trait.FromWasmAbi.html#impl-FromWasmAbi-for-Box%3C%5BJsValue%5D%3E
        history: Vec<JsValue>,
        documents: Vec<JsValue>,
    ) -> Result<Vec<u32>, JsValue> {
        let history = history
            .iter()
            .map(JsValue::into_serde)
            .collect::<Result<Vec<DocumentHistory>, _>>()
            .into_js_result()?;
        let documents = documents
            .iter()
            .map(JsValue::into_serde)
            .collect::<Result<Vec<Document>, _>>()
            .into_js_result()?;
        let reranked = self.0.rerank(&history, &documents);

        let ranks = reranked
            .into_iter()
            .map(|(id, rank)| (id, rank as u32))
            .collect::<HashMap<_, _>>();
        documents
            .iter()
            .map(|document| ranks.get(&document.id).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                JsValue::from_str(
                    "Failed to rerank the documents: The document ids are inconsistent",
                )
            })
    }

    pub fn serialize(&self) -> Result<Uint8Array, JsValue> {
        self.0
            .serialize()
            .into_js_result()
            .map(|bytes| bytes.as_slice().into())
    }

    // See [`xaynai_faults()`] for more.
    // pub fn faults(&self) -> Faults {

    //     self.0.errors().into()
    // }

    // /// See [`xaynai_analytics()`] for more.
    // unsafe fn analytics(xaynai: *const Self) -> Result<CAnalytics, ExternError> {
    //     let xaynai = unsafe { xaynai.as_ref() }.ok_or_else(|| {
    //         CCode::AiPointer.with_context("Failed to get the analytics: The ai pointer is null")
    //     })?;

    //     Ok(CAnalytics(xaynai.0.analytics().cloned()))
    // }
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
