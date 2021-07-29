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
    SMBertVocabPointer,
    /// A smbert model null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    SMBertModelPointer,
    /// A qambert vocab null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    QAMBertVocabPointer,
    /// A qambert model null pointer error.
    #[allow(clippy::upper_case_acronyms)]
    QAMBertModelPointer,
    /// A LTR model null pointer error.
    LtrModelPointer,
    /// A vocab or model file IO error.
    ReadFile,
    /// A Xayn AI initialization error.
    InitAi,
    /// A Xayn AI null pointer error.
    AiPointer,
    /// A document histories null pointer error.
    HistoriesPointer,
    /// A document history id null pointer error.
    HistoryIdPointer,
    /// A document history session id null pointer error.
    HistorySessionPointer,
    /// A document history query id null pointer error.
    HistoryQueryIdPointer,
    /// A document history query words null pointer error.
    HistoryQueryWordsPointer,
    /// A document history url null pointer error.
    HistoryUrlPointer,
    /// A document history domain null pointer error.
    HistoryDomainPointer,
    /// A documents null pointer error.
    DocumentsPointer,
    /// A document id null pointer error.
    DocumentIdPointer,
    /// A document title null pointer error.
    DocumentTitlePointer,
    /// A document snippet null pointer error.
    DocumentSnippetPointer,
    /// A document session id null pointer error.
    DocumentSessionPointer,
    /// A document query id null pointer error.
    DocumentQueryIdPointer,
    /// A document query words null pointer error.
    DocumentQueryWordsPointer,
    /// A document url null pointer error.
    DocumentUrlPointer,
    /// A document domain null pointer error.
    DocumentDomainPointer,
    /// Deserialization of reranker database error.
    RerankerDeserialization,
    /// Serialization of reranker database error.
    RerankerSerialization,
    /// Deserialization of history collection error.
    HistoriesDeserialization,
    /// Deserialization of document collection error.
    DocumentsDeserialization,
    /// Deserialization of rerank mode error.
    RerankModeDeserialization,
    /// Serialization of sync data error.
    SyncDataSerialization,
    /// Synchronization error.
    Synchronization,
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
