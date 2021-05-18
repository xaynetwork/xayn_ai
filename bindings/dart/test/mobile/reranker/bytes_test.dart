import 'dart:ffi' show nullptr;
import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/mobile/reranker/bytes.dart' show Bytes;

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

    test('free', () {
      final bytes =
          Bytes.fromList(Uint8List.fromList(List.generate(10, (i) => i)));
      expect(bytes.ptr, isNot(equals(nullptr)));
      bytes.free();
      expect(bytes.ptr, equals(nullptr));
    });
  });
}
