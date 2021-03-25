import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show malloc, Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/ai.dart' show ffi;
import 'package:xayn_ai_ffi_dart/ffi.dart' show CXaynAiCode, CXaynAiError;

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

  /// A Bert builder error.
  buildBert,

  /// A Reranker builder error.
  buildReranker,

  /// A Xayn AI null pointer error.
  xaynAiPointer,

  /// A documents null pointer error.
  documentsPointer,

  /// A document id null pointer error.
  idPointer,

  /// A document snippet null pointer error.
  snippetPointer,
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
      case XaynAiCode.buildBert:
        return CXaynAiCode.BuildBert;
      case XaynAiCode.buildReranker:
        return CXaynAiCode.BuildReranker;
      case XaynAiCode.xaynAiPointer:
        return CXaynAiCode.XaynAiPointer;
      case XaynAiCode.documentsPointer:
        return CXaynAiCode.DocumentsPointer;
      case XaynAiCode.idPointer:
        return CXaynAiCode.IdPointer;
      case XaynAiCode.snippetPointer:
        return CXaynAiCode.SnippetPointer;
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
      case CXaynAiCode.BuildBert:
        return XaynAiCode.buildBert;
      case CXaynAiCode.BuildReranker:
        return XaynAiCode.buildReranker;
      case CXaynAiCode.XaynAiPointer:
        return XaynAiCode.xaynAiPointer;
      case CXaynAiCode.DocumentsPointer:
        return XaynAiCode.documentsPointer;
      case CXaynAiCode.IdPointer:
        return XaynAiCode.idPointer;
      case CXaynAiCode.SnippetPointer:
        return XaynAiCode.snippetPointer;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

/// The Xayn AI FFI error information.
class XaynAiError {
  late Pointer<CXaynAiError> _error;

  /// Gets the pointer.
  Pointer<CXaynAiError> get ptr => _error;

  /// Creates the error information initialized to success.
  XaynAiError() {
    _error = malloc.call<CXaynAiError>();
    _error.ref.code = CXaynAiCode.Success;
    _error.ref.message = nullptr;
  }

  /// Checks for a panic code.
  bool isPanic() => _error.ref.code == CXaynAiCode.Panic;

  /// Checks for a success code.
  bool isSuccess() => _error.ref.code == CXaynAiCode.Success;

  /// Checks for an error code.
  bool isError() => !(isPanic() || isSuccess());

  /// Creates an exception from the error information.
  XaynAiException toException() {
    final code = XaynAiCodeInt.fromInt(_error.ref.code);
    final message =
        isSuccess() ? '' : _error.ref.message.cast<Utf8>().toDartString();
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

  @override
  String toString() => '${_code.toString().split('.').last}: $_message';
}
