use js_sys::Uint8Array;
use wasm_bindgen::prelude::{wasm_bindgen, JsValue};
use xayn_ai::{Builder, Document, DocumentHistory, RerankMode, Reranker};
use xayn_ai_ffi::CCode;

use crate::error::IntoJsResult;

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
        smbert_vocab: &[u8],
        smbert_model: &[u8],
        qambert_vocab: &[u8],
        qambert_model: &[u8],
        ltr_model: &[u8],
        serialized: Option<Box<[u8]>>,
    ) -> Result<WXaynAi, JsValue> {
        console_error_panic_hook::set_once();

        Builder::default()
            .with_serialized_database(serialized)
            .map_err(|cause| {
                CCode::RerankerDeserialization.with_context(format!(
                    "Failed to deserialize the reranker database: {}",
                    cause,
                ))
            })
            .into_js_result()?
            .with_smbert_from_reader(smbert_vocab, smbert_model)
            .with_qambert_from_reader(qambert_vocab, qambert_model)
            .with_domain_from_reader(ltr_model)
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
        mode: JsValue,
        histories: Vec<JsValue>,
        documents: Vec<JsValue>,
    ) -> Result<JsValue, JsValue> {
        let mode = mode
            .into_serde::<RerankMode>()
            .map_err(|cause| {
                CCode::RerankModeDeserialization
                    .with_context(format!("Failed to deserialize the rerank mode: {}", cause))
            })
            .into_js_result()?;
        let histories = histories
            .iter()
            .map(|js_value| js_value.into_serde::<DocumentHistory>().map(Into::into))
            .collect::<Result<Vec<DocumentHistory>, _>>()
            .map_err(|cause| {
                CCode::HistoriesDeserialization.with_context(format!(
                    "Failed to deserialize the collection of histories: {}",
                    cause,
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
                    cause,
                ))
            })
            .into_js_result()?;

        let outcomes = self.0.rerank(mode, &histories, &documents);

        Ok(JsValue::from_serde(&outcomes).expect("Failed to serialize the analytics"))
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
    use xayn_ai::{
        DayOfWeek,
        DocumentHistory,
        DocumentId,
        QueryId,
        Relevance,
        RerankMode,
        SessionId,
        UserAction,
        UserFeedback,
    };
    use xayn_ai_ffi::Error;

    /// Path to the current smbert vocabulary file.
    const SMBERT_VOCAB: &[u8] = include_bytes!("../../data/smbert_v0000/vocab.txt");

    /// Path to the current smbert onnx model file.
    const SMBERT_MODEL: &[u8] = include_bytes!("../../data/smbert_v0000/smbert.onnx");

    /// Path to the current qambert vocabulary file.
    const QAMBERT_VOCAB: &[u8] = include_bytes!("../../data/qambert_v0001/vocab.txt");

    /// Path to the current qambert onnx model file.
    const QAMBERT_MODEL: &[u8] = include_bytes!("../../data/qambert_v0001/qambert.onnx");

    /// Path to the current ltr model binparams file.
    const LTR_MODEL: &[u8] = include_bytes!("../../data/ltr_v0000/ltr.binparams");

    impl std::fmt::Debug for WXaynAi {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            f.debug_struct("WXaynAi").finish()
        }
    }

    fn test_histories() -> Vec<JsValue> {
        let len = 6;

        let ids = (0..len)
            .map(|idx| DocumentId::from_u128(idx as u128))
            .collect::<Vec<_>>();
        let relevances = repeat(Relevance::Low.into())
            .take(len / 2)
            .chain(repeat(Relevance::High.into()).take(len - len / 2));
        let feedbacks = repeat(UserFeedback::Irrelevant.into())
            .take(len / 2)
            .chain(repeat(UserFeedback::Relevant.into()).take(len - len / 2));
        let sessions = (0..len)
            .map(|idx| SessionId::from_u128(idx as u128))
            .collect::<Vec<_>>();
        let query_counts = repeat(1).take(len);
        let query_ids = (0..len)
            .map(|idx| QueryId::from_u128(idx as u128))
            .collect::<Vec<_>>();
        let query_words = (0..len)
            .map(|idx| format!("query {}", idx))
            .collect::<Vec<_>>();
        let days = repeat(DayOfWeek::Sun)
            .take(len / 2)
            .chain(repeat(DayOfWeek::Mon).take(len - len / 2));
        let urls = (0..len)
            .map(|idx| format!("url-{}", idx))
            .collect::<Vec<_>>();
        let domains = (0..len)
            .map(|idx| format!("domain-{}", idx))
            .collect::<Vec<_>>();
        let ranks = 0..len;
        let user_actions = repeat(UserAction::Miss)
            .take(len / 2)
            .chain(repeat(UserAction::Click).take(len - len / 2));

        izip!(
            ids,
            relevances,
            feedbacks,
            sessions,
            query_counts,
            query_ids,
            query_words,
            days,
            urls,
            domains,
            ranks,
            user_actions
        )
        .map(|hist| {
            JsValue::from_serde(&DocumentHistory {
                id: hist.0,
                relevance: hist.1,
                user_feedback: hist.2,
                session: hist.3,
                query_count: hist.4,
                query_id: hist.5,
                query_words: hist.6,
                day: hist.7,
                url: hist.8,
                domain: hist.9,
                rank: hist.10,
                user_action: hist.11,
            })
            .unwrap()
        })
        .collect()
    }

    fn test_documents() -> Vec<JsValue> {
        let len = 10;

        let ids = (0..len)
            .map(|idx| DocumentId::from_u128(idx as u128))
            .collect::<Vec<_>>();
        let titles = (0..len)
            .map(|idx| format!("title {}", idx))
            .collect::<Vec<_>>();
        let snippets = (0..len)
            .map(|idx| format!("snippet {}", idx))
            .collect::<Vec<_>>();
        let ranks = 0..len as usize;
        let sessions = (0..len)
            .map(|idx| SessionId::from_u128(idx as u128))
            .collect::<Vec<_>>();
        let query_counts = repeat(1).take(len);
        let query_ids = (0..len)
            .map(|idx| QueryId::from_u128(idx as u128))
            .collect::<Vec<_>>();
        let query_words = (0..len)
            .map(|idx| format!("query {}", idx))
            .collect::<Vec<_>>();
        let urls = (0..len)
            .map(|idx| format!("url-{}", idx))
            .collect::<Vec<_>>();
        let domains = (0..len)
            .map(|idx| format!("domain-{}", idx))
            .collect::<Vec<_>>();

        izip!(
            ids,
            titles,
            snippets,
            ranks,
            sessions,
            query_counts,
            query_ids,
            query_words,
            urls,
            domains
        )
        .map(|doc| {
            JsValue::from_serde(&Document {
                id: doc.0,
                title: doc.1,
                snippet: doc.2,
                rank: doc.3,
                session: doc.4,
                query_count: doc.5,
                query_id: doc.6,
                query_words: doc.7,
                url: doc.8,
                domain: doc.9,
            })
            .unwrap()
        })
        .collect()
    }

    fn rerank_mode_search() -> JsValue {
        JsValue::from_serde(&RerankMode::Search).expect("rerank mode search")
    }

    #[wasm_bindgen_test]
    fn test_rerank() {
        let mut xaynai = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap();
        xaynai
            .rerank(rerank_mode_search(), test_histories(), test_documents())
            .unwrap();
    }

    #[wasm_bindgen_test]
    fn test_serialize() {
        let xaynai = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap();
        xaynai.serialize().unwrap();
    }

    #[wasm_bindgen_test]
    fn test_faults() {
        let xaynai = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap();
        let faults = xaynai.faults();
        assert!(faults.is_empty());
    }

    #[wasm_bindgen_test]
    fn test_analytics() {
        let xaynai = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap();
        let analytics = xaynai.analytics();
        assert!(analytics.is_null());
    }

    #[wasm_bindgen_test]
    fn test_smbert_vocab_empty() {
        let error = WXaynAi::new(
            &[],
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap_err()
        .into_serde::<Error>()
        .unwrap();

        assert_eq!(error.code(), CCode::InitAi);
        assert_eq!(
            error.message(),
            "Failed to initialize the ai: Failed to build the tokenizer: Failed to build the tokenizer: Failed to build the model: Missing any entry in the vocabulary",
        );
    }

    #[wasm_bindgen_test]
    fn test_qambert_vocab_empty() {
        let error = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            &[],
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap_err()
        .into_serde::<Error>()
        .unwrap();

        assert_eq!(error.code(), CCode::InitAi);
        assert_eq!(
            error.message(),
            "Failed to initialize the ai: Failed to build the tokenizer: Failed to build the tokenizer: Failed to build the model: Missing any entry in the vocabulary",
        );
    }

    #[wasm_bindgen_test]
    fn test_smbert_model_empty() {
        let error = WXaynAi::new(
            SMBERT_VOCAB,
            &[],
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap_err()
        .into_serde::<Error>()
        .unwrap();

        assert_eq!(error.code(), CCode::InitAi);
        assert_eq!(
            error.message(),
            "Failed to initialize the ai: Failed to build the model: Failed to run a tract operation: model proto does not contain a graph",
        );
    }

    #[wasm_bindgen_test]
    fn test_qambert_model_empty() {
        let error = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            &[],
            LTR_MODEL,
            None,
        )
        .unwrap_err()
        .into_serde::<Error>()
        .unwrap();

        assert_eq!(error.code(), CCode::InitAi);
        assert_eq!(
            error.message(),
            "Failed to initialize the ai: Failed to build the model: Failed to run a tract operation: model proto does not contain a graph",
        );
    }

    #[wasm_bindgen_test]
    fn test_ltr_model_empty() {
        let error = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            &[],
            None,
        )
        .unwrap_err()
        .into_serde::<Error>()
        .unwrap();

        assert_eq!(error.code(), CCode::InitAi);
        assert_eq!(
            error.message(),
            "Failed to initialize the ai: io error: failed to fill whole buffer",
        );
    }

    #[wasm_bindgen_test]
    fn test_smbert_model_invalid() {
        let error = WXaynAi::new(
            SMBERT_VOCAB,
            &[0],
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap_err()
        .into_serde::<Error>()
        .unwrap();

        assert_eq!(error.code(), CCode::InitAi);
        assert!(error
            .message()
            .contains("Failed to initialize the ai: Failed to build the model: "));
    }

    #[wasm_bindgen_test]
    fn test_qambert_model_invalid() {
        let error = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            &[0],
            LTR_MODEL,
            None,
        )
        .unwrap_err()
        .into_serde::<Error>()
        .unwrap();

        assert_eq!(error.code(), CCode::InitAi);
        assert!(error
            .message()
            .contains("Failed to initialize the ai: Failed to build the model: "));
    }

    #[wasm_bindgen_test]
    fn test_ltr_model_invalid() {
        let error = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            &[0],
            None,
        )
        .unwrap_err()
        .into_serde::<Error>()
        .unwrap();

        assert_eq!(error.code(), CCode::InitAi);
        assert_eq!(
            error.message(),
            "Failed to initialize the ai: io error: failed to fill whole buffer"
        );
    }

    #[wasm_bindgen_test]
    fn test_history_invalid() {
        let mut xaynai = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap();
        let error = xaynai
            .rerank(
                rerank_mode_search(),
                vec![JsValue::from("invalid")],
                test_documents(),
            )
            .unwrap_err()
            .into_serde::<Error>()
            .unwrap();

        assert_eq!(error.code(), CCode::HistoriesDeserialization);
        assert!(error
            .message()
            .contains("Failed to deserialize the collection of histories: invalid type: "));
    }

    #[wasm_bindgen_test]
    fn test_history_empty() {
        let mut xaynai = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap();
        xaynai
            .rerank(rerank_mode_search(), vec![], test_documents())
            .unwrap();
    }

    #[wasm_bindgen_test]
    fn test_documents_invalid() {
        let mut xaynai = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap();
        let error = xaynai
            .rerank(
                rerank_mode_search(),
                test_histories(),
                vec![JsValue::from("invalid")],
            )
            .unwrap_err()
            .into_serde::<Error>()
            .unwrap();

        assert_eq!(error.code(), CCode::DocumentsDeserialization);
        assert!(error
            .message()
            .contains("Failed to deserialize the collection of documents: invalid type: "));
    }

    #[wasm_bindgen_test]
    fn test_documents_empty() {
        let mut xaynai = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            None,
        )
        .unwrap();
        xaynai
            .rerank(rerank_mode_search(), test_histories(), vec![])
            .unwrap();
    }

    #[wasm_bindgen_test]
    fn test_serialized_empty() {
        WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            Some(Box::new([])),
        )
        .unwrap();
    }

    #[wasm_bindgen_test]
    fn test_serialized_invalid() {
        let error = WXaynAi::new(
            SMBERT_VOCAB,
            SMBERT_MODEL,
            QAMBERT_VOCAB,
            QAMBERT_MODEL,
            LTR_MODEL,
            Some(Box::new([1, 2, 3])),
        )
        .unwrap_err()
        .into_serde::<Error>()
        .unwrap();

        assert_eq!(error.code(), CCode::RerankerDeserialization);
        assert!(error.message().contains(
            "Failed to deserialize the reranker database: Unsupported serialized data. "
        ));
    }
}
