import 'dart:ffi' show nullptr, StructPointer;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/src/mobile/data/document.dart' show Documents;
import '../utils.dart' show documents, mkTestDoc;

void main() {
  group('Document', () {
    test('empty', () {
      expect(() => mkTestDoc('', 'abc', 0), throwsArgumentError);
      expect(
        () => mkTestDoc('fcb6a685-eb92-4d36-8686-000000000000', 'abc', -1),
        throwsArgumentError,
      );
    });
  });

  group('Documents', () {
    test('new', () {
      final docs = Documents(documents);
      documents.asMap().forEach((i, document) {
        var doc = docs.ptr.ref.data[i];
        expect(
          doc.id.cast<Utf8>().toDartString(),
          equals(document.id),
        );
        expect(
          doc.title.cast<Utf8>().toDartString(),
          equals(document.title),
        );
        expect(
          doc.rank,
          equals(document.rank),
        );
        expect(
          doc.session.cast<Utf8>().toDartString(),
          equals(document.session),
        );
        expect(
          doc.query_count,
          equals(document.queryCount),
        );
        expect(
          doc.query_id.cast<Utf8>().toDartString(),
          equals(document.queryId),
        );
        expect(
          doc.query_words.cast<Utf8>().toDartString(),
          equals(document.queryWords),
        );
        expect(
          doc.url.cast<Utf8>().toDartString(),
          equals(document.url),
        );
        expect(
          doc.domain.cast<Utf8>().toDartString(),
          equals(document.domain),
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
