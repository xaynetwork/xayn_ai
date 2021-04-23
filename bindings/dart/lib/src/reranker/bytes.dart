import 'dart:ffi' show nullptr, Pointer, StructPointer, Uint8Pointer;
import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CBytes;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;
import 'package:xayn_ai_ffi_dart/src/result/error.dart' show XaynAiError;

/// A bytes buffer.
class Bytes {
  late Pointer<CBytes> _bytes;

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
    } finally {
      error.free();
    }

    bytes.asMap().forEach((i, byte) {
      _bytes.ref.data[i] = byte;
    });
  }

  /// Gets the pointer.
  Pointer<CBytes> get ptr => _bytes;

  /// Converts the buffer to a list.
  Uint8List toList() {
    if (_bytes == nullptr || _bytes.ref.data == nullptr) {
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
    if (_bytes != nullptr) {
      ffi.bytes_drop(_bytes);
      _bytes = nullptr;
    }
  }
}
