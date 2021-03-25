import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show malloc, Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/ai.dart' show ffi;
import 'package:xayn_ai_ffi_dart/ffi.dart' show CXaynAiError, CXaynAiErrorCode;

/// The Xayn AI FFI error information.
class XaynAiError {
  late Pointer<CXaynAiError> _error;

  /// Gets the pointer.
  Pointer<CXaynAiError> get ptr => _error;

  /// Creates the error information initialized to success.
  XaynAiError() {
    _error = malloc.call<CXaynAiError>();
    _error.ref.code = CXaynAiErrorCode.Success;
    _error.ref.message = nullptr;
  }

  /// Checks for a panic code.
  bool isPanic() => _error.ref.code == CXaynAiErrorCode.Panic;

  /// Checks for a success code.
  bool isSuccess() => _error.ref.code == CXaynAiErrorCode.Success;

  /// Checks for an error code.
  bool isError() => !(isPanic() || isSuccess());

  /// Gets the error/panic message.
  @override
  String toString() {
    if (isSuccess()) {
      return '';
    }

    final message = _error.ref.message.cast<Utf8>().toDartString();
    switch (_error.ref.code) {
      case CXaynAiErrorCode.Panic:
        return 'Panic: $message';
      case CXaynAiErrorCode.VocabPointer:
        return 'VocabPointer: $message';
      case CXaynAiErrorCode.ModelPointer:
        return 'ModelPointer: $message';
      case CXaynAiErrorCode.ReadFile:
        return 'ReadFile: $message';
      case CXaynAiErrorCode.BuildBert:
        return 'BuildBert: $message';
      case CXaynAiErrorCode.BuildReranker:
        return 'BuildReranker: $message';
      case CXaynAiErrorCode.XaynAiPointer:
        return 'XaynAiPointer: $message';
      case CXaynAiErrorCode.DocumentsPointer:
        return 'DocumentsPointer: $message';
      case CXaynAiErrorCode.IdPointer:
        return 'IdPointer: $message';
      case CXaynAiErrorCode.SnippetPointer:
        return 'SnippetPointer: $message';
      default:
        throw UnsupportedError('Undefined error code');
    }
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
  late final String _message;

  /// Creates a Xayn Ai exception.
  XaynAiException(_message);

  @override
  String toString() => _message;
}
