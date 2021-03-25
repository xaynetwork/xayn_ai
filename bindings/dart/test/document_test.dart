import 'dart:ffi' show nullptr, StructPointer;

import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/document.dart' show Documents;

void main() {
  const ids = ['0', '1', '2'];
  const snippets = ['abc', 'def', 'ghi'];
  const ranks = [0, 1, 2];

  group('Documents', () {
    test('new', () {
      final docs = Documents(ids, snippets, ranks);

      expect(docs.ptr, isNot(equals(nullptr)));
      expect(docs.size, equals(ranks.length));
      expect(docs.ranks, equals(ranks));

      docs.free();
    });

    test('ranks', () {
      final docs = Documents(ids, snippets, ranks);
      final reranks = List.from(ranks.reversed, growable: false);
      for (var i = 0; i < docs.size; i++) {
        docs.ptr[i].rank = reranks[i];
      }

      expect(docs.ranks, equals(reranks));

      docs.free();
    });

    test('double free', () {
      final docs = Documents(ids, snippets, ranks);
      docs.free();
      docs.free();
    });

    test('invalid size', () {
      expect(() => Documents([], snippets, ranks), throwsArgumentError);
      expect(() => Documents(ids, [], ranks), throwsArgumentError);
      expect(() => Documents(ids, snippets, []), throwsArgumentError);
    });

    test('invalid ranks', () {
      final docs = Documents(ids, snippets, ranks);
      docs.free();
      expect(() => docs.ranks, throwsArgumentError);
    });
  });
}
