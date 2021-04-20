import 'dart:ffi' show nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CWarnings;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

/// The Xayn Ai warnings.
class Warnings {
  Pointer<CWarnings> _warns;

  /// Creates the warnings.
  ///
  /// This constructor never throws an exception.
  Warnings(this._warns);

  /// Converts the warnings to a list.
  List<String> toList() =>
      _warns == nullptr || _warns.ref.data == nullptr || _warns.ref.len == 0
          ? List.empty()
          : List.generate(
              _warns.ref.len,
              (i) => _warns.ref.data[i].message == nullptr
                  ? ''
                  : _warns.ref.data[i].message.cast<Utf8>().toDartString(),
              growable: false,
            );

  /// Frees the memory.
  void free() {
    if (_warns != nullptr) {
      ffi.warnings_drop(_warns);
      _warns = nullptr;
    }
  }
}
