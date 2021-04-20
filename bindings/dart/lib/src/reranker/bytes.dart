import 'dart:ffi'
    show AllocatorAlloc, nullptr, Pointer, StructPointer, Uint8, Uint8Pointer;
import 'dart:typed_data' show Uint8List;

import 'package:ffi/ffi.dart' show malloc;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CBytes;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

/// A bytes buffer.
class Bytes {
  late Pointer<CBytes> _bytes;
  final bool _owned;

  /// Creates the borrowed bytes buffer.
  ///
  /// This constructor never throws an exception.
  Bytes(this._bytes) : _owned = false;

  /// Creates the owned bytes buffer.
  ///
  /// This constructor never throws an exception.
  Bytes.fromList(Uint8List bytes) : _owned = true {
    _bytes = malloc.call<CBytes>();
    _bytes.ref.len = bytes.length;
    if (bytes.isEmpty) {
      _bytes.ref.data = nullptr;
    } else {
      _bytes.ref.data = malloc.call<Uint8>(_bytes.ref.len);
      bytes.asMap().forEach((i, byte) {
        _bytes.ref.data[i] = byte;
      });
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
      if (_owned) {
        if (_bytes.ref.data != nullptr) {
          malloc.free(_bytes.ref.data);
        }
        malloc.free(_bytes);
      } else {
        ffi.bytes_drop(_bytes);
      }
      _bytes = nullptr;
    }
  }
}
