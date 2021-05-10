/// The Xayn AI error codes.
enum Code {
  /// A warning or noncritical error.
  fault,

  /// An irrecoverable error.
  panic,

  /// No error.
  none,

  /// A vocab null pointer error.
  vocabPointer,

  /// A model null pointer error.
  modelPointer,

  /// A vocab or model file IO error.
  readFile,

  /// A Xayn AI initialization error.
  initAi,

  /// A Xayn AI null pointer error.
  aiPointer,

  /// A document histories null pointer error.
  historiesPointer,

  /// A document history id null pointer error.
  historyIdPointer,

  /// A documents null pointer error.
  documentsPointer,

  /// A document id null pointer error.
  documentIdPointer,

  /// A document snippet null pointer error.
  documentSnippetPointer,

  /// Deserialization of reranker database error.
  rerankerDeserialization,

  /// Serialization of reranker database error.
  rerankerSerialization,
}

/// A Xayn AI exception.
class XaynAiException implements Exception {
  final Code code;
  final String message;

  /// Creates a Xayn AI exception.
  const XaynAiException(this.code, this.message);
}
