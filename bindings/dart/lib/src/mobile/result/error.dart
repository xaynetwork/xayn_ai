import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show malloc, Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/src/common/ffi/genesis.dart' show CCode;
import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show IntToCode, XaynAiException;
import 'package:xayn_ai_ffi_dart/src/common/utils.dart' show assertNeq;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart' show CError;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/library.dart' show ffi;

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
