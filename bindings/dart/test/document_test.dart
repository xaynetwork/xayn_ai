import 'dart:ffi' show nullptr, StructPointer;

import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/document.dart' show Documents;

void main() {
  group('Documents', () {
    test('new', () {
      final ranks = [0, 1, 2];
      final docs = Documents(['0', '1', '2'], ['abc', 'def', 'ghi'], ranks);

      expect(docs.ptr, isNot(equals(nullptr)));
      expect(docs.size, equals(ranks.length));
      expect(docs.ranks, equals(ranks));

      docs.free();
    });

    test('ranks', () {
      final ranks = [0, 1, 2];
      final docs = Documents(['0', '1', '2'], ['abc', 'def', 'ghi'], ranks);

      expect(docs.ptr, isNot(equals(nullptr)));
      final reranks = List.from(ranks.reversed, growable: false);
      for (var i = 0; i < docs.size; i++) {
        docs.ptr[i].rank = reranks[i];
      }
      expect(docs.ranks, equals(reranks));

      docs.free();
    });

    test('error size', () {
      expect(() => Documents([], [], []), throwsArgumentError);
      expect(() => Documents(['0'], ['abc', 'def'], [0]), throwsArgumentError);
      expect(() => Documents(['0'], ['abc'], [0, 1]), throwsArgumentError);
    });

    test('error ranks', () {
      final docs = Documents(['0'], ['abc'], [0]);
      docs.free();
      expect(() => docs.ranks, throwsArgumentError);
    });
  });
}
