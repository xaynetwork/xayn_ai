import 'dart:ffi'
    show
        AllocatorAlloc,
        nullptr,
        StructPointer,
        Uint8,
        // ignore: unused_shown_name
        Uint8Pointer;
import 'dart:typed_data' show Uint8List;

import 'package:ffi/ffi.dart' show malloc;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CBytes;
import 'package:xayn_ai_ffi_dart/src/reranker/bytes.dart' show Bytes;

void main() {
  group('Bytes', () {
    test('to list', () {
      final bytes = Uint8List.fromList(List.generate(10, (i) => i));
      final borrowed = malloc.call<CBytes>();
      borrowed.ref.data = malloc.call<Uint8>(bytes.length);
      borrowed.ref.len = bytes.length;
      bytes.asMap().forEach((i, byte) => borrowed.ref.data[i] = byte);
      expect(Bytes(borrowed).toList(), equals(bytes));
      malloc.free(borrowed.ref.data);
      malloc.free(borrowed);
    });

    test('null', () {
      final bytes = Bytes(nullptr);
      expect(bytes.toList(), isEmpty);
    });

    test('empty', () {
      final borrowed = malloc.call<CBytes>();
      borrowed.ref.data = nullptr;
      borrowed.ref.len = 0;
      expect(Bytes(borrowed).toList(), isEmpty);
      malloc.free(borrowed.ref.data);
      malloc.free(borrowed);
    });
  });

  group('Bytes owned', () {
    test('to list', () {
      final bytes = Uint8List.fromList(List.generate(10, (i) => i));
      final owned = Bytes.fromList(bytes);
      expect(owned.toList(), equals(bytes));
      owned.free();
    });

    test('empty', () {
      final bytes = Uint8List(0);
      final owned = Bytes.fromList(bytes);
      expect(owned.toList(), isEmpty);
      owned.free();
    });

    test('free', () {
      final bytes = Uint8List.fromList(List.generate(10, (i) => i));
      final owned = Bytes.fromList(bytes);
      expect(owned.ptr, isNot(equals(nullptr)));
      owned.free();
      expect(owned.ptr, equals(nullptr));
    });
  });
}
