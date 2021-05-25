import 'dart:ffi' show nullptr, Uint16Pointer, FloatPointer;

import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CBoxedSlice_f32, CBoxedSlice_u16;

extension BoxedSliceU16List on CBoxedSlice_u16 {
  /// Converts a `CBoxedSlice<u16>` to an `Uint16List`.
  List<int>? toList() {
    if (data == nullptr) {
      return null;
    } else if (len == 0) {
      return List.empty();
    } else {
      return data.asTypedList(len).toList(growable: false);
    }
  }
}

extension BoxedSliceF32List on CBoxedSlice_f32 {
  /// Converts a `CBoxedSlice<f32>` to a `Float32List`.
  List<double>? toList() {
    if (data == nullptr) {
      return null;
    } else if (len == 0) {
      return List.empty();
    } else {
      return data.asTypedList(len).toList(growable: false);
    }
  }
}
