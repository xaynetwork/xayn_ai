import 'dart:ffi' show AllocatorAlloc, Int8, nullptr, StructPointer;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer, Utf8;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, test;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CError, CWarnings;
import 'package:xayn_ai_ffi_dart/src/result/warning.dart' show Warnings;

void main() {
  group('Warnings', () {
    test('to list', () {
      final warnings = List.generate(10, (i) => 'warning $i', growable: false);
      final warningsPtr = malloc.call<CWarnings>();
      warningsPtr.ref.data = malloc.call<CError>(warnings.length);
      warningsPtr.ref.len = warnings.length;
      warnings.asMap().forEach((i, warning) => warningsPtr.ref.data[i].message =
          warning.toNativeUtf8().cast<Int8>());
      expect(Warnings(warningsPtr).toList(), equals(warnings));
      for (var i = 0; i < warnings.length; i++) {
        malloc.free(warningsPtr.ref.data[i].message.cast<Utf8>());
      }
      malloc.free(warningsPtr.ref.data);
      malloc.free(warningsPtr);
    });

    test('null', () {
      final warnings = Warnings(nullptr);
      expect(warnings.toList(), isEmpty);
    });

    test('empty', () {
      final warningsPtr = malloc.call<CWarnings>();
      warningsPtr.ref.data = nullptr;
      warningsPtr.ref.len = 0;
      expect(Warnings(warningsPtr).toList(), isEmpty);
      malloc.free(warningsPtr.ref.data);
      malloc.free(warningsPtr);
    });
  });
}
