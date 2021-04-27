use serde::Serialize;
use wasm_bindgen::JsValue;

use crate::utils::IntoJsResult;

// placeholder / later we can have a crate that contains common code for c-ffi and wasm
#[repr(i32)]
pub enum CCode {
    /// A warning or uncritical error.
    Fault = -2,
    /// A Xayn AI initialization error.
    InitAi = 4,
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
    code: i32,
    message: String,
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
