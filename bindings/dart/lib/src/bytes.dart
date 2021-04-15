import 'dart:ffi' show nullptr, Pointer, StructPointer, Uint8Pointer;
import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CBytes;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

/// An array of bytes.
class Bytes {
  Pointer<CBytes> _array;

  /// Creates the array.
  Bytes(this._array);

  /// Converts the array to a list.
  Uint8List toList() {
    final len = _array.ref.len;
    final bytes = Uint8List(len);

    // ptr is never read if the array is empty
    final ptr = _array.ref.ptr;
    for (var i = 0; i < len; i++) {
      bytes[i] = ptr[i];
    }

    return bytes;
  }

  /// Frees the memory.
  void free() {
    if (_array != nullptr) {
      ffi.bytes_drop(_array);
      _array = nullptr;
    }
  }
}
