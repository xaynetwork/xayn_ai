import 'dart:ffi' show nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CBoxedSlice_CError;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;
import 'package:xayn_ai_ffi_dart/src/utils.dart' show assertEq, assertNeq;

/// The Xayn Ai faults.
class Faults {
  Pointer<CBoxedSlice_CError> _faults;

  /// Creates the faults.
  ///
  /// This constructor never throws an exception.
  Faults(this._faults);

  /// Converts the faults to a list.
  List<String> toList() {
    assert(
      _faults == nullptr || _faults.ref.data != nullptr,
      'unexpected faults pointer state',
    );

    return _faults == nullptr
        ? List.empty()
        : List.generate(
            _faults.ref.len,
            (i) {
              if (_faults.ref.data[i].message == nullptr) {
                return '';
              } else {
                assertNeq(_faults.ref.data[i].message.ref.data, nullptr);
                assertEq(
                  _faults.ref.data[i].message.ref.len,
                  _faults.ref.data[i].message.ref.data.cast<Utf8>().length + 1,
                );

                return _faults.ref.data[i].message.ref.data
                    .cast<Utf8>()
                    .toDartString();
              }
            },
            growable: false,
          );
  }

  /// Frees the memory.
  void free() {
    if (_faults != nullptr) {
      ffi.faults_drop(_faults);
      _faults = nullptr;
    }
  }
}
