import 'dart:ffi' show nullptr, Uint16Pointer, FloatPointer;
import 'dart:typed_data' show Uint16List, Float32List;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart'
    show CBoxedSlice_f32, CBoxedSlice_u16;

extension BoxedSliceU16List on CBoxedSlice_u16 {
  /// Converts a `CBoxedSlice<u16>` to an `Uint16List`.
  Uint16List toList() {
    if (data == nullptr || len == 0) {
      return Uint16List(0);
    } else {
      final result = Uint16List(len);
      data.asTypedList(len).asMap().forEach((i, item) {
        result[i] = item;
      });
      return result;
    }
  }
}

extension BoxedSliceF32List on CBoxedSlice_f32 {
  /// Converts a `CBoxedSlice<f32>` to a `Float32List`.
  Float32List toList() {
    if (data == nullptr || len == 0) {
      return Float32List(0);
    } else {
      final result = Float32List(len);
      data.asTypedList(len).asMap().forEach((i, item) {
        result[i] = item;
      });
      return result;
    }
  }
}
