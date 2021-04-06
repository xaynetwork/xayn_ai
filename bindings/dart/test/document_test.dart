import 'dart:ffi' show AllocatorAlloc, nullptr, Uint32;

import 'package:ffi/ffi.dart' show malloc;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/document.dart' show Documents, History, Ranks;
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
  });

  group('Ranks', () {
    test('to list', () {
      final size = docsRanks.length;
      final ranksPtr = malloc.call<Uint32>(size);
      final ranks = Ranks(ranksPtr, size);

      expect(ranks.toList().length, equals(size));

      malloc.free(ranksPtr);
    });

    test('empty', () {
      final ranks = Ranks(nullptr, 0);

      expect(ranks.toList(), isEmpty);
    });

    test('double free', () {
      final ranks = Ranks(nullptr, 0);

      ranks.free();
    });
  });
}
