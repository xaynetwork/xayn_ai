import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show malloc, Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/ffi.dart' show CXaynAiCode, CXaynAiError;
import 'package:xayn_ai_ffi_dart/library.dart' show ffi;

/// The Xayn AI error codes.
enum XaynAiCode {
  /// An irrecoverable error.
  panic,

  /// No error.
  success,

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

  /// A document history null pointer error.
  historyPointer,

  /// A document history id null pointer error.
  historyIdPointer,

  /// A documents null pointer error.
  documentsPointer,

  /// A document id null pointer error.
  documentIdPointer,

  /// A document snippet null pointer error.
  documentSnippetPointer,

  /// An internal error.
  internal,
}

extension XaynAiCodeInt on XaynAiCode {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case XaynAiCode.panic:
        return CXaynAiCode.Panic;
      case XaynAiCode.success:
        return CXaynAiCode.Success;
      case XaynAiCode.vocabPointer:
        return CXaynAiCode.VocabPointer;
      case XaynAiCode.modelPointer:
        return CXaynAiCode.ModelPointer;
      case XaynAiCode.readFile:
        return CXaynAiCode.ReadFile;
      case XaynAiCode.initAi:
        return CXaynAiCode.InitAi;
      case XaynAiCode.aiPointer:
        return CXaynAiCode.AiPointer;
      case XaynAiCode.historyPointer:
        return CXaynAiCode.HistoryPointer;
      case XaynAiCode.historyIdPointer:
        return CXaynAiCode.HistoryIdPointer;
      case XaynAiCode.documentsPointer:
        return CXaynAiCode.DocumentsPointer;
      case XaynAiCode.documentIdPointer:
        return CXaynAiCode.DocumentIdPointer;
      case XaynAiCode.documentSnippetPointer:
        return CXaynAiCode.DocumentSnippetPointer;
      case XaynAiCode.internal:
        return CXaynAiCode.Internal;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }

  /// Creates the error code from a discriminant.
  static XaynAiCode fromInt(int idx) {
    switch (idx) {
      case CXaynAiCode.Panic:
        return XaynAiCode.panic;
      case CXaynAiCode.Success:
        return XaynAiCode.success;
      case CXaynAiCode.VocabPointer:
        return XaynAiCode.vocabPointer;
      case CXaynAiCode.ModelPointer:
        return XaynAiCode.modelPointer;
      case CXaynAiCode.ReadFile:
        return XaynAiCode.readFile;
      case CXaynAiCode.InitAi:
        return XaynAiCode.initAi;
      case CXaynAiCode.AiPointer:
        return XaynAiCode.aiPointer;
      case CXaynAiCode.HistoryPointer:
        return XaynAiCode.historyPointer;
      case CXaynAiCode.HistoryIdPointer:
        return XaynAiCode.historyIdPointer;
      case CXaynAiCode.DocumentsPointer:
        return XaynAiCode.documentsPointer;
      case CXaynAiCode.DocumentIdPointer:
        return XaynAiCode.documentIdPointer;
      case CXaynAiCode.DocumentSnippetPointer:
        return XaynAiCode.documentSnippetPointer;
      case CXaynAiCode.Internal:
        return XaynAiCode.internal;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

/// The Xayn AI error information.
class XaynAiError {
  late Pointer<CXaynAiError> _error;

  /// Creates the error information initialized to success.
  XaynAiError() {
    _error = malloc.call<CXaynAiError>();
    _error.ref.code = CXaynAiCode.Success;
    _error.ref.message = nullptr;
  }

  /// Gets the pointer.
  Pointer<CXaynAiError> get ptr => _error;

  /// Checks for an irrecoverable error code.
  bool isPanic() => _error.ref.code == CXaynAiCode.Panic;

  /// Checks for a success code.
  bool isSuccess() => _error.ref.code == CXaynAiCode.Success;

  /// Checks for an error code (both recoverable and irrecoverable).
  bool isError() => !isSuccess();

  /// Creates an exception from the error information.
  XaynAiException toException() {
    final code = XaynAiCodeInt.fromInt(_error.ref.code);
    final message = _error.ref.message == nullptr
        ? ''
        : _error.ref.message.cast<Utf8>().toDartString();
    return XaynAiException(code, message);
  }

  /// Frees the memory.
  void free() {
    if (_error != nullptr) {
      ffi.error_message_drop(_error);
      malloc.free(_error);
      _error = nullptr;
    }
  }
}

/// A Xayn AI exception.
class XaynAiException implements Exception {
  final XaynAiCode _code;
  final String _message;

  /// Creates a Xayn AI exception.
  const XaynAiException(this._code, this._message);

  /// Gets the code.
  XaynAiCode get code => _code;

  /// Gets the message.
  @override
  String toString() => _message;
}
