use std::{any::Any, convert::Infallible};

use derive_more::Display;
use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

/// The Xayn AI error codes.
#[repr(i8)]
#[derive(Clone, Copy, Debug, Deserialize_repr, Display, PartialEq, Serialize_repr)]
pub enum CCode {
    /// A warning or noncritical error.
    Fault = -2,
    /// An irrecoverable error.
    Panic = -1,
    /// No error.
    None = 0,
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
    pub fn with_context(self, message: impl Into<String>) -> Error {
        Error {
            code: self,
            message: message.into(),
        }
    }
}

/// The Xayn AI error information.
#[derive(Debug, Deserialize, Serialize)]
pub struct Error {
    code: CCode,
    message: String,
}

impl Error {
    /// Gets the error code.
    pub fn code(&self) -> CCode {
        self.code
    }

    /// Gets the error message.
    pub fn message(&self) -> &str {
        self.message.as_str()
    }

    /// Creates the error information for the no error code.
    pub fn none() -> Self {
        CCode::None.with_context(String::new())
    }

    /// Creates the error information from the panic payload.
    pub fn panic(payload: Box<dyn Any + Send + 'static>) -> Self {
        // https://doc.rust-lang.org/std/panic/struct.PanicInfo.html#method.payload
        let message = if let Some(message) = payload.downcast_ref::<&str>() {
            message
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message
        } else {
            "Unknown panic"
        };

        CCode::Panic.with_context(message)
    }
}

impl From<Infallible> for Error {
    fn from(_none: Infallible) -> Self {
        Self::none()
    }
}
