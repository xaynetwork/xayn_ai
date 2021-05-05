import 'dart:ffi' show AllocatorAlloc, nullptr, StructPointer, Uint8;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer, Utf8, Utf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, test;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart'
    show CError, CBoxedSlice_CError, CBoxedSlice_u8;
import 'package:xayn_ai_ffi_dart/src/result/fault.dart' show Faults;

void main() {
  group('Faults', () {
    test('to list', () {
      final faults = List.generate(10, (i) => 'fault $i', growable: false);
      final faultsPtr = malloc.call<CBoxedSlice_CError>();
      faultsPtr.ref.data = malloc.call<CError>(faults.length);
      faultsPtr.ref.len = faults.length;
      faults.asMap().forEach((i, fault) {
        faultsPtr.ref.data[i].message = malloc.call<CBoxedSlice_u8>();
        faultsPtr.ref.data[i].message.ref.data =
            fault.toNativeUtf8().cast<Uint8>();
        faultsPtr.ref.data[i].message.ref.len =
            faultsPtr.ref.data[i].message.ref.data.cast<Utf8>().length + 1;
      });
      expect(Faults(faultsPtr).toList(), equals(faults));
      for (var i = 0; i < faults.length; i++) {
        malloc.free(faultsPtr.ref.data[i].message.ref.data);
        malloc.free(faultsPtr.ref.data[i].message);
      }
      malloc.free(faultsPtr.ref.data);
      malloc.free(faultsPtr);
    });

    test('null', () {
      final faults = Faults(nullptr);
      expect(faults.toList(), isEmpty);
    });

    test('empty', () {
      final faultsPtr = malloc.call<CBoxedSlice_CError>();
      faultsPtr.ref.data = nullptr;
      faultsPtr.ref.len = 0;
      expect(Faults(faultsPtr).toList(), isEmpty);
      malloc.free(faultsPtr.ref.data);
      malloc.free(faultsPtr);
    });
  });
}
