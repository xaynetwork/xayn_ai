import 'dart:ffi' show nullptr, StructPointer;
import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/reranker/bytes.dart' show Bytes;

void main() {
  group('Bytes', () {
    test('list', () {
      final list = Uint8List.fromList(List.generate(10, (i) => i));
      final bytes = Bytes.fromList(list);
      expect(bytes.toList(), equals(list));
      bytes.free();
    });

    test('null', () {
      final bytes = Bytes(nullptr);
      expect(bytes.toList(), isEmpty);
    });

    test('empty', () {
      final bytes = Bytes.fromList(Uint8List(0));
      expect(bytes.toList(), isEmpty);
      bytes.free();
    });

    test('invalid data', () {
      final len = 10;
      final bytes = Bytes.fromList(Uint8List(len));
      bytes.ptr.ref.len = 0;
      expect(bytes.toList(), isEmpty);
      bytes.ptr.ref.len = len;
      bytes.free();
    });

    test('invalid len', () {
      final bytes = Bytes.fromList(Uint8List(0));
      bytes.ptr.ref.len = 10;
      expect(bytes.toList(), isEmpty);
    });

    test('free', () {
      final bytes =
          Bytes.fromList(Uint8List.fromList(List.generate(10, (i) => i)));
      expect(bytes.ptr, isNot(equals(nullptr)));
      bytes.free();
      expect(bytes.ptr, equals(nullptr));
    });
  });
}
