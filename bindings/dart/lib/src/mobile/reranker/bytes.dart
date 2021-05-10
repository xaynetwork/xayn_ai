import 'dart:ffi' show nullptr, Pointer, StructPointer, Uint8Pointer;
import 'dart:typed_data' show Uint8List;

import 'package:flutter/foundation.dart' show listEquals;

import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CBoxedSlice_u8;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/library.dart' show ffi;
import 'package:xayn_ai_ffi_dart/src/mobile/result/error.dart' show XaynAiError;
import 'package:xayn_ai_ffi_dart/src/utils.dart' show assertNeq;

/// A bytes buffer.
class Bytes {
  late Pointer<CBoxedSlice_u8> _bytes;

  /// Creates the bytes buffer from a pointer.
  ///
  /// This constructor never throws an exception.
  Bytes(this._bytes);

  /// Creates the bytes buffer from a list.
  ///
  /// This constructor can throw an exception.
  Bytes.fromList(Uint8List bytes) {
    final error = XaynAiError();

    _bytes = ffi.bytes_new(bytes.length, error.ptr);
    try {
      if (error.isError()) {
        throw error.toException();
      }
      assertNeq(_bytes, nullptr);
      assert(listEquals(
        _bytes.ref.data.asTypedList(_bytes.ref.len),
        Uint8List(bytes.length),
      ));
    } finally {
      error.free();
    }

    bytes.asMap().forEach((i, byte) {
      _bytes.ref.data[i] = byte;
    });
  }

  /// Gets the pointer.
  Pointer<CBoxedSlice_u8> get ptr => _bytes;

  /// Converts the buffer to a list.
  Uint8List toList() {
    assert(
      _bytes == nullptr || _bytes.ref.data != nullptr,
      'unexpected bytes pointer state',
    );

    if (_bytes == nullptr) {
      return Uint8List(0);
    } else {
      final bytes = Uint8List(_bytes.ref.len);
      _bytes.ref.data.asTypedList(_bytes.ref.len).asMap().forEach((i, byte) {
        bytes[i] = byte;
      });
      return bytes;
    }
  }

  /// Frees the memory.
  void free() {
    assert(
      _bytes == nullptr || _bytes.ref.data != nullptr,
      'unexpected bytes pointer state',
    );

    if (_bytes != nullptr) {
      ffi.bytes_drop(_bytes);
      _bytes = nullptr;
    }
  }
}
