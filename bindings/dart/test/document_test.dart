import 'dart:ffi' show nullptr;

import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/document.dart' show Documents, History;
import 'utils.dart'
    show
        docsIds,
        docsRanks,
        docsSnippets,
        histFeedbacks,
        histIds,
        histRelevances;

void main() {
  group('History', () {
    test('new', () {
      final hist = History(histIds, histRelevances, histFeedbacks);
      expect(hist.ptr, isNot(equals(nullptr)));
      expect(hist.size, equals(histIds.length));
      expect(hist.size, equals(histRelevances.length));
      expect(hist.size, equals(histFeedbacks.length));
      hist.free();
    });

    test('empty', () {
      final hist = History([], [], []);
      expect(hist.ptr, equals(nullptr));
      expect(hist.size, 0);
    });

    test('double free', () {
      final hist = History(histIds, histRelevances, histFeedbacks);
      expect(hist.ptr, isNot(equals(nullptr)));
      hist.free();
      expect(hist.ptr, equals(nullptr));
      hist.free();
      expect(hist.ptr, equals(nullptr));
    });

    test('invalid size', () {
      expect(() => History([], histRelevances, histFeedbacks),
          throwsArgumentError);
      expect(() => History(histIds, [], histFeedbacks), throwsArgumentError);
      expect(() => History(histIds, histRelevances, []), throwsArgumentError);
    });
  });

  group('Documents', () {
    test('new', () {
      final docs = Documents(docsIds, docsSnippets, docsRanks);
      expect(docs.ptr, isNot(equals(nullptr)));
      expect(docs.size, equals(docsIds.length));
      expect(docs.size, equals(docsSnippets.length));
      expect(docs.size, equals(docsRanks.length));
      docs.free();
    });

    test('empty', () {
      final docs = Documents([], [], []);
      expect(docs.ptr, equals(nullptr));
      expect(docs.size, 0);
    });

    test('ranks', () {
      final docs = Documents(docsIds, docsSnippets, docsRanks);
      expect(docs.ranks, equals(docsRanks));
      docs.free();
    });

    test('double free', () {
      final docs = Documents(docsIds, docsSnippets, docsRanks);
      expect(docs.ptr, isNot(equals(nullptr)));
      docs.free();
      expect(docs.ptr, equals(nullptr));
      docs.free();
      expect(docs.ptr, equals(nullptr));
    });

    test('invalid size', () {
      expect(() => Documents([], docsSnippets, docsRanks), throwsArgumentError);
      expect(() => Documents(docsIds, [], docsRanks), throwsArgumentError);
      expect(() => Documents(docsIds, docsSnippets, []), throwsArgumentError);
    });

    test('invalid ranks', () {
      final docs = Documents(docsIds, docsSnippets, docsRanks);
      docs.free();
      expect(() => docs.ranks, throwsArgumentError);
    });
  });
}
