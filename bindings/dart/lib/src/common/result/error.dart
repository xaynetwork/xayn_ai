import 'package:meta/meta.dart' show immutable;

import 'package:xayn_ai_ffi_dart/src/common/ffi/genesis.dart' show CCode;

/// The Xayn AI error codes.
enum Code {
  /// A warning or noncritical error.
  fault,

  /// An irrecoverable error.
  panic,

  /// No error.
  none,

  /// A smbert vocab null pointer error.
  smbertVocabPointer,

  /// A smbert model null pointer error.
  smbertModelPointer,

  /// A qambert vocab null pointer error.
  qambertVocabPointer,

  /// A qambert model null pointer error.
  qambertModelPointer,

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

  /// A document history session id null pointer error.
  historySessionPointer,

  /// A document history query id null pointer error.
  historyQueryIdPointer,

  /// A document history query words null pointer error.
  historyQueryWordsPointer,

  /// A document history url null pointer error.
  historyUrlPointer,

  /// A document history domain null pointer error.
  historyDomainPointer,

  /// A documents null pointer error.
  documentsPointer,

  /// A document id null pointer error.
  documentIdPointer,

  /// A document title null pointer error.
  documentTitlePointer,

  /// A document snippet null pointer error.
  documentSnippetPointer,

  /// A document session id null pointer error.
  documentSessionPointer,

  /// A document query id null pointer error.
  documentQueryIdPointer,

  /// A document query words null pointer error.
  documentQueryWordsPointer,

  /// A document url null pointer error.
  documentUrlPointer,

  /// A document domain null pointer error.
  documentDomainPointer,

  /// Deserialization of reranker database error.
  rerankerDeserialization,

  /// Serialization of reranker database error.
  rerankerSerialization,

  /// Deserialization of history collection error.
  historiesDeserialization,

  /// Deserialization of document collection error.
  documentsDeserialization,
}

extension CodeToInt on Code {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Code.fault:
        return CCode.Fault;
      case Code.panic:
        return CCode.Panic;
      case Code.none:
        return CCode.None;
      case Code.smbertVocabPointer:
        return CCode.SMBertVocabPointer;
      case Code.smbertModelPointer:
        return CCode.SMBertModelPointer;
      case Code.qambertVocabPointer:
        return CCode.QAMBertVocabPointer;
      case Code.qambertModelPointer:
        return CCode.QAMBertModelPointer;
      case Code.readFile:
        return CCode.ReadFile;
      case Code.initAi:
        return CCode.InitAi;
      case Code.aiPointer:
        return CCode.AiPointer;
      case Code.historiesPointer:
        return CCode.HistoriesPointer;
      case Code.historyIdPointer:
        return CCode.HistoryIdPointer;
      case Code.historySessionPointer:
        return CCode.HistorySessionPointer;
      case Code.historyQueryIdPointer:
        return CCode.HistoryQueryIdPointer;
      case Code.historyQueryWordsPointer:
        return CCode.HistoryQueryWordsPointer;
      case Code.historyUrlPointer:
        return CCode.HistoryUrlPointer;
      case Code.historyDomainPointer:
        return CCode.HistoryDomainPointer;
      case Code.documentsPointer:
        return CCode.DocumentsPointer;
      case Code.documentIdPointer:
        return CCode.DocumentIdPointer;
      case Code.documentTitlePointer:
        return CCode.DocumentTitlePointer;
      case Code.documentSnippetPointer:
        return CCode.DocumentSnippetPointer;
      case Code.documentSessionPointer:
        return CCode.DocumentSessionPointer;
      case Code.documentQueryIdPointer:
        return CCode.DocumentQueryIdPointer;
      case Code.documentQueryWordsPointer:
        return CCode.DocumentQueryWordsPointer;
      case Code.documentUrlPointer:
        return CCode.DocumentUrlPointer;
      case Code.documentDomainPointer:
        return CCode.DocumentDomainPointer;
      case Code.rerankerDeserialization:
        return CCode.RerankerDeserialization;
      case Code.rerankerSerialization:
        return CCode.RerankerSerialization;
      case Code.historiesDeserialization:
        return CCode.HistoriesDeserialization;
      case Code.documentsDeserialization:
        return CCode.DocumentsDeserialization;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

extension IntToCode on int {
  /// Creates the error code from a discriminant.
  Code toCode() {
    switch (this) {
      case CCode.Fault:
        return Code.fault;
      case CCode.Panic:
        return Code.panic;
      case CCode.None:
        return Code.none;
      case CCode.SMBertVocabPointer:
        return Code.smbertVocabPointer;
      case CCode.SMBertModelPointer:
        return Code.smbertModelPointer;
      case CCode.QAMBertVocabPointer:
        return Code.qambertVocabPointer;
      case CCode.QAMBertModelPointer:
        return Code.qambertModelPointer;
      case CCode.ReadFile:
        return Code.readFile;
      case CCode.InitAi:
        return Code.initAi;
      case CCode.AiPointer:
        return Code.aiPointer;
      case CCode.HistoriesPointer:
        return Code.historiesPointer;
      case CCode.HistoryIdPointer:
        return Code.historyIdPointer;
      case CCode.HistorySessionPointer:
        return Code.historySessionPointer;
      case CCode.HistoryQueryIdPointer:
        return Code.historyQueryIdPointer;
      case CCode.HistoryQueryWordsPointer:
        return Code.historyQueryWordsPointer;
      case CCode.HistoryUrlPointer:
        return Code.historyUrlPointer;
      case CCode.HistoryDomainPointer:
        return Code.historyDomainPointer;
      case CCode.DocumentsPointer:
        return Code.documentsPointer;
      case CCode.DocumentIdPointer:
        return Code.documentIdPointer;
      case CCode.DocumentTitlePointer:
        return Code.documentTitlePointer;
      case CCode.DocumentSnippetPointer:
        return Code.documentSnippetPointer;
      case CCode.DocumentSessionPointer:
        return Code.documentSessionPointer;
      case CCode.DocumentQueryIdPointer:
        return Code.documentQueryIdPointer;
      case CCode.DocumentQueryWordsPointer:
        return Code.documentQueryWordsPointer;
      case CCode.DocumentUrlPointer:
        return Code.documentUrlPointer;
      case CCode.DocumentDomainPointer:
        return Code.documentDomainPointer;
      case CCode.RerankerDeserialization:
        return Code.rerankerDeserialization;
      case CCode.RerankerSerialization:
        return Code.rerankerSerialization;
      case CCode.HistoriesDeserialization:
        return Code.historiesDeserialization;
      case CCode.DocumentsDeserialization:
        return Code.documentsDeserialization;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

/// A Xayn AI exception.
@immutable
class XaynAiException implements Exception {
  final Code code;
  final String message;

  /// Creates a Xayn AI exception.
  const XaynAiException(this.code, this.message);

  @override
  String toString() => message;
}
