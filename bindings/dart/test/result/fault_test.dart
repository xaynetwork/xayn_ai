import 'dart:ffi' show AllocatorAlloc, Int8, nullptr, StructPointer;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer, Utf8;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, test;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CError, CFaults;
import 'package:xayn_ai_ffi_dart/src/result/fault.dart' show Faults;

void main() {
  group('Faults', () {
    test('to list', () {
      final faults = List.generate(10, (i) => 'fault $i', growable: false);
      final faultsPtr = malloc.call<CFaults>();
      faultsPtr.ref.data = malloc.call<CError>(faults.length);
      faultsPtr.ref.len = faults.length;
      faults.asMap().forEach((i, fault) =>
          faultsPtr.ref.data[i].message = fault.toNativeUtf8().cast<Int8>());
      expect(Faults(faultsPtr).toList(), equals(faults));
      for (var i = 0; i < faults.length; i++) {
        malloc.free(faultsPtr.ref.data[i].message.cast<Utf8>());
      }
      malloc.free(faultsPtr.ref.data);
      malloc.free(faultsPtr);
    });

    test('null', () {
      final faults = Faults(nullptr);
      expect(faults.toList(), isEmpty);
    });

    test('empty', () {
      final faultsPtr = malloc.call<CFaults>();
      faultsPtr.ref.data = nullptr;
      faultsPtr.ref.len = 0;
      expect(Faults(faultsPtr).toList(), isEmpty);
      malloc.free(faultsPtr.ref.data);
      malloc.free(faultsPtr);
    });
  });
}
