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
    /// A smbert vocab null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    SMBertVocabPointer = 1,
    /// A smbert model null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    SMBertModelPointer = 2,
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
    /// A document history session id null pointer error.
    HistorySessionPointer = 8,
    /// A document history query id null pointer error.
    HistoryQueryIdPointer = 9,
    /// A document history query words null pointer error.
    HistoryQueryWordsPointer = 10,
    /// A document history url null pointer error.
    HistoryUrlPointer = 11,
    /// A document history domain null pointer error.
    HistoryDomainPointer = 12,
    /// A documents null pointer error.
    DocumentsPointer = 13,
    /// A document id null pointer error.
    DocumentIdPointer = 14,
    /// A document snippet null pointer error.
    DocumentSnippetPointer = 15,
    /// A document session id null pointer error.
    DocumentSessionPointer = 16,
    /// A document query id null pointer error.
    DocumentQueryIdPointer = 17,
    /// A document query words null pointer error.
    DocumentQueryWordsPointer = 18,
    /// A document url null pointer error.
    DocumentUrlPointer = 19,
    /// A document domain null pointer error.
    DocumentDomainPointer = 20,
    /// Deserialization of reranker database error.
    RerankerDeserialization = 21,
    /// Serialization of reranker database error.
    RerankerSerialization = 22,
    /// A qambert vocab null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    QAMBertVocabPointer = 23,
    /// A qambert model null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    QAMBertModelPointer = 24,
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
