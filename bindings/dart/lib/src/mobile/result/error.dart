import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show malloc, Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show Code, XaynAiException;
import 'package:xayn_ai_ffi_dart/src/common/utils.dart' show assertNeq;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CCode, CError;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/library.dart' show ffi;

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
      case Code.vocabPointer:
        return CCode.VocabPointer;
      case Code.modelPointer:
        return CCode.ModelPointer;
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
      case Code.documentsPointer:
        return CCode.DocumentsPointer;
      case Code.documentIdPointer:
        return CCode.DocumentIdPointer;
      case Code.documentSnippetPointer:
        return CCode.DocumentSnippetPointer;
      case Code.rerankerDeserialization:
        return CCode.RerankerDeserialization;
      case Code.rerankerSerialization:
        return CCode.RerankerSerialization;
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
      case CCode.VocabPointer:
        return Code.vocabPointer;
      case CCode.ModelPointer:
        return Code.modelPointer;
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
      case CCode.DocumentsPointer:
        return Code.documentsPointer;
      case CCode.DocumentIdPointer:
        return Code.documentIdPointer;
      case CCode.DocumentSnippetPointer:
        return Code.documentSnippetPointer;
      case CCode.RerankerDeserialization:
        return Code.rerankerDeserialization;
      case CCode.RerankerSerialization:
        return Code.rerankerSerialization;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

/// The Xayn AI error information.
class XaynAiError {
  late Pointer<CError> _error;

  /// Creates the error information initialized to success.
  ///
  /// This constructor never throws an exception.
  XaynAiError() {
    _error = malloc.call<CError>();
    _error.ref.code = CCode.None;
    _error.ref.message = nullptr;
  }

  /// Gets the pointer.
  Pointer<CError> get ptr => _error;

  /// Checks for a fault code.
  bool isFault() {
    assertNeq(_error, nullptr);
    return _error.ref.code == CCode.Fault;
  }

  /// Checks for an irrecoverable error code.
  bool isPanic() {
    assertNeq(_error, nullptr);
    return _error.ref.code == CCode.Panic;
  }

  /// Checks for a no error code.
  bool isNone() {
    assertNeq(_error, nullptr);
    return _error.ref.code == CCode.None;
  }

  /// Checks for an error code (both recoverable and irrecoverable).
  bool isError() => !isNone() && !isFault();

  /// Creates an exception from the error information.
  XaynAiException toException() {
    assertNeq(_error, nullptr);
    assert(
      _error.ref.message == nullptr ||
          (_error.ref.message.ref.data != nullptr &&
              _error.ref.message.ref.len ==
                  _error.ref.message.ref.data.cast<Utf8>().length + 1),
      'unexpected error pointer state',
    );

    final code = _error.ref.code.toCode();
    final message = _error.ref.message == nullptr
        ? ''
        : _error.ref.message.ref.data.cast<Utf8>().toDartString();

    return XaynAiException(code, message);
  }

  /// Frees the memory.
  void free() {
    assert(
      _error == nullptr ||
          _error.ref.message == nullptr ||
          (_error.ref.message.ref.data != nullptr &&
              _error.ref.message.ref.len ==
                  _error.ref.message.ref.data.cast<Utf8>().length + 1),
      'unexpected error pointer state',
    );

    if (_error != nullptr) {
      ffi.error_message_drop(_error);
      malloc.free(_error);
      _error = nullptr;
    }
  }
}
