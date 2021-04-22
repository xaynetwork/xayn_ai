use std::collections::HashMap;

use js_sys::Uint8Array;
use wasm_bindgen::prelude::{wasm_bindgen, JsValue};
use xayn_ai::{Builder, Document, DocumentHistory, Reranker};

use super::utils::ToJsResult;

#[wasm_bindgen]
pub struct WXaynAi(Reranker);

#[wasm_bindgen]
pub fn xaynai_new(
    vocab: &[u8],
    model: &[u8],
    serialized: Option<Box<[u8]>>,
) -> Result<WXaynAi, JsValue> {
    Builder::default()
        .with_serialized_database(&serialized.unwrap_or_default())
        .to_js_result()?
        .with_bert_from_reader(vocab, model)
        .build()
        .map(WXaynAi)
        .to_js_result()
}

#[wasm_bindgen]
pub fn xaynai_rerank(
    xaynai: &mut WXaynAi,
    // vec is the same as Box<[JsValue]>
    // https://rustwasm.github.io/wasm-bindgen/api/wasm_bindgen/convert/trait.FromWasmAbi.html#impl-FromWasmAbi-for-Box%3C%5BJsValue%5D%3E
    history: Vec<JsValue>,
    documents: Vec<JsValue>,
) -> Result<Vec<u32>, JsValue> {
    // https://rustwasm.github.io/docs/wasm-bindgen/reference/arbitrary-data-with-serde.html
    let history = history
        .iter()
        .map(|h| h.into_serde())
        .collect::<Result<Vec<DocumentHistory>, _>>()
        .to_js_result()?;
    let documents = documents
        .iter()
        .map(|d| d.into_serde())
        .collect::<Result<Vec<Document>, _>>()
        .to_js_result()?;
    let reranked = xaynai.0.rerank(&history, &documents);

    let ranks = reranked
        .into_iter()
        .map(|(id, rank)| (id, rank as u32))
        .collect::<HashMap<_, _>>();
    documents
        .iter()
        .map(|document| ranks.get(&document.id).copied())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            JsValue::from_str("Failed to rerank the documents: The document ids are inconsistent")
        })
}

#[wasm_bindgen]
pub fn xaynai_serialize(xaynai: &WXaynAi) -> Result<Uint8Array, JsValue> {
    // https://github.com/rustwasm/wasm-bindgen/issues/2402#issuecomment-754032667
    xaynai
        .0
        .serialize()
        .to_js_result()
        .map(|bytes| bytes.as_slice().into())
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
        let mut xaynai = xaynai_new(VOCAB, MODEL, None).unwrap();

        let document = Document {
            id: "1".into(),
            rank: 0,
            snippet: "abc".to_string(),
        };
        let js_document = JsValue::from_serde(&document).unwrap();

        let ranks = xaynai_rerank(&mut xaynai, vec![], vec![js_document]).unwrap();
        assert_eq!(ranks, [0]);

        let ser = xaynai_serialize(&xaynai).unwrap();
        assert!(ser.length() != 0)
    }
}
