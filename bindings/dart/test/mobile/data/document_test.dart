import 'dart:ffi' show nullptr, StructPointer;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/mobile/data/document.dart' show Documents;
import '../utils.dart' show documents;

void main() {
  group('Document', () {
    test('empty', () {
      final id = 'fcb6a685-eb92-4d36-8686-000000000000';
      expect(
        () => Document(
          id: '', // empty id
          title: 'title',
          snippet: 'snippet',
          rank: 0,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          url: 'url',
          domain: 'domain',
          viewed: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => Document(
          id: id,
          title: 'title',
          snippet: 'snippet',
          rank: -1, // negative rank
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          url: 'url',
          domain: 'domain',
          viewed: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => Document(
          id: id,
          title: 'title',
          snippet: 'snippet',
          rank: 0,
          session: '', // empty session id
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          url: 'url',
          domain: 'domain',
          viewed: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => Document(
          id: id,
          title: 'title',
          snippet: 'snippet',
          rank: 0,
          session: id,
          queryCount: -1, // negative query count
          queryId: id,
          queryWords: 'query words',
          url: 'url',
          domain: 'domain',
          viewed: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => Document(
          id: id,
          title: 'title',
          snippet: 'snippet',
          rank: 0,
          session: id,
          queryCount: 1,
          queryId: '', // empty query id
          queryWords: 'query words',
          url: 'url',
          domain: 'domain',
          viewed: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => Document(
          id: id,
          title: 'title',
          snippet: 'snippet',
          rank: 0,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: '', // empty query words
          url: 'url',
          domain: 'domain',
          viewed: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => Document(
          id: id,
          title: 'title',
          snippet: 'snippet',
          rank: 0,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          url: '', // empty url
          domain: 'domain',
          viewed: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => Document(
          id: id,
          title: 'title',
          snippet: 'snippet',
          rank: 0,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          url: 'url',
          domain: '', // empty domain
          viewed: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => Document(
          id: id,
          title: 'title',
          snippet: 'snippet',
          rank: 0,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          url: 'url',
          domain: 'domain',
          viewed: -1, // negative viewed
        ),
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
        expect(doc.viewed, equals(document.viewed));
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
