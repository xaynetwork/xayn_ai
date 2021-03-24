import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show malloc, Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/ai.dart' show xaynAiFfi;
import 'package:xayn_ai_ffi_dart/ffi.dart' show CXaynAiError, CXaynAiErrorCode;

/// The Xayn AI error information.
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

    return _error.ref.message.cast<Utf8>().toDartString();
  }

  /// Frees the memory.
  void free() {
    if (_error != nullptr) {
      xaynAiFfi.error_message_drop(_error);
      malloc.free(_error);
      _error = nullptr;
    }
  }
}

class XaynAiException implements Exception {
  late final String _message;

  XaynAiException(_message);

  @override
  String toString() => _message;
}
