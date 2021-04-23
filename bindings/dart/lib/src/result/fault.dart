import 'dart:ffi' show nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CFaults;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

/// The Xayn Ai faults.
class Faults {
  Pointer<CFaults> _faults;

  /// Creates the faults.
  ///
  /// This constructor never throws an exception.
  Faults(this._faults);

  /// Converts the faults to a list.
  List<String> toList() =>
      _faults == nullptr || _faults.ref.data == nullptr || _faults.ref.len == 0
          ? List.empty()
          : List.generate(
              _faults.ref.len,
              (i) => _faults.ref.data[i].message == nullptr
                  ? ''
                  : _faults.ref.data[i].message.cast<Utf8>().toDartString(),
              growable: false,
            );

  /// Frees the memory.
  void free() {
    if (_faults != nullptr) {
      ffi.faults_drop(_faults);
      _faults = nullptr;
    }
  }
}
