import 'dart:ffi'
    show AllocatorAlloc, nullptr, Pointer, StructPointer, Uint8, Uint8Pointer;
import 'dart:typed_data' show Uint8List;

import 'package:ffi/ffi.dart' show malloc;

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
    if (bytes.isEmpty) {
      _bytes = ffi.bytes_new(nullptr, 0, error.ptr);
    } else {
      final bytesPtr = malloc.call<Uint8>(bytes.length);
      bytes.asMap().forEach((i, byte) {
        bytesPtr[i] = byte;
      });
      _bytes = ffi.bytes_new(bytesPtr, bytes.length, error.ptr);
      malloc.free(bytesPtr);
    }
    if (error.isError()) {
      throw error.toException();
    }
  }

  /// Gets the pointer.
  Pointer<CBytes> get ptr => _bytes;

  /// Converts the buffer to a list.
  Uint8List toList() {
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
    if (_bytes != nullptr) {
      ffi.bytes_drop(_bytes);
      _bytes = nullptr;
    }
  }
}
