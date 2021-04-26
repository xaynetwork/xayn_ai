use serde::Serialize;
use wasm_bindgen::JsValue;

use crate::utils::IntoJsResult;

// just a placeholder
#[repr(i32)]
pub enum CCode {
    /// A warning or uncritical error.
    Fault = -2,
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
    /// A Xayn AI initialization error.
    InitAi = 4,
    /// A Xayn AI null pointer error.
    AiPointer = 5,
    /// A document histories null pointer error.
    HistoriesPointer = 6,
    /// A document history id null pointer error.
    HistoryIdPointer = 7,
    /// A documents null pointer error.
    DocumentsPointer = 8,
    /// A document id null pointer error.
    DocumentIdPointer = 9,
    /// A document snippet null pointer error.
    DocumentSnippetPointer = 10,
    /// Deserialization of reranker database error.
    RerankerDeserialization = 11,
    /// Serialization of reranker database error.
    RerankerSerialization = 12,
    /// Deserialization of history collection error.
    HistoriesDeserialization = 13,
    /// Deserialization of document collection error.
    DocumentsDeserialization = 14,
}

impl CCode {
    /// Provides context for the error code.
    pub fn with_context(self, message: impl Into<String>) -> ExternError {
        ExternError::new_error(self as i32, message)
    }
}

#[derive(Serialize)]
pub struct ExternError {
    pub code: i32,
    pub message: String,
}

impl ExternError {
    fn new_error(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl<T> IntoJsResult<T> for Result<T, ExternError> {
    fn into_js_result(self) -> Result<T, JsValue> {
        self.map_err(|e| JsValue::from_serde(&e).unwrap())
    }
}
