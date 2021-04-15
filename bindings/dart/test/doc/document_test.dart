import 'dart:ffi' show nullptr, StructPointer;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/src/doc/document.dart'
    show Document, Documents;
import '../utils.dart' show documents;

void main() {
  group('Document', () {
    test('empty', () {
      expect(() => Document('', 'abc', 0), throwsArgumentError);
      expect(() => Document('0', '', 0), throwsArgumentError);
      expect(() => Document('0', 'abc', -1), throwsArgumentError);
    });
  });

  group('Documents', () {
    test('new', () {
      final docs = Documents(documents);
      documents.asMap().forEach((i, document) {
        expect(
          docs.ptr.ref.data[i].id.cast<Utf8>().toDartString(),
          equals(document.id),
        );
        expect(
          docs.ptr.ref.data[i].snippet.cast<Utf8>().toDartString(),
          equals(document.snippet),
        );
        expect(
          docs.ptr.ref.data[i].rank,
          equals(document.rank),
        );
      });
      expect(docs.ptr.ref.len, equals(documents.length));
      docs.free();
    });

    test('empty', () {
      final docs = Documents([]);
      expect(docs.ptr.ref.data, equals(nullptr));
      expect(docs.ptr.ref.len, equals(0));
      docs.free();
    });

    test('free', () {
      final docs = Documents(documents);
      expect(docs.ptr, isNot(equals(nullptr)));
      docs.free();
      expect(docs.ptr, equals(nullptr));
    });
  });
}
