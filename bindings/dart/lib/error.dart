import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show calloc, Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/ai.dart' show xaynAiFfi;
import 'package:xayn_ai_ffi_dart/ffi.dart' show CXaynAiError, CXaynAiErrorCode;

class XaynAiError {
  late final Pointer<CXaynAiError> _error;

  /// Gets the pointer.
  Pointer<CXaynAiError> get ptr => _error;

  /// Creates an error handler.
  XaynAiError() {
    _error = calloc.call<CXaynAiError>();
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
  ///
  /// Nulls the pointer afterwards to prevent double-frees.
  void free() {
    if (_error != nullptr) {
      xaynAiFfi.error_message_drop(_error);
      calloc.free(_error);
      _error = nullptr;
    }
  }
}
