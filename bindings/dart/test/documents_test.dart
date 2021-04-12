import 'dart:ffi' show AllocatorAlloc, nullptr, Uint32;

import 'package:ffi/ffi.dart' show malloc;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/doc/documents.dart'
    show Documents, Histories, Ranks;
import 'utils.dart' show documents, histories;

void main() {
  group('Histories', () {
    test('new', () {
      final hists = Histories(histories);
      expect(hists.ptr, isNot(equals(nullptr)));
      expect(hists.size, equals(histories.length));
      hists.free();
    });

    test('empty', () {
      final hists = Histories([]);
      expect(hists.ptr, equals(nullptr));
      expect(hists.size, equals(0));
    });

    test('free', () {
      final hist = Histories(histories);
      expect(hist.ptr, isNot(equals(nullptr)));
      hist.free();
      expect(hist.ptr, equals(nullptr));
    });
  });

  group('Documents', () {
    test('new', () {
      final docs = Documents(documents);
      expect(docs.ptr, isNot(equals(nullptr)));
      expect(docs.size, equals(documents.length));
      docs.free();
    });

    test('empty', () {
      final docs = Documents([]);
      expect(docs.ptr, equals(nullptr));
      expect(docs.size, equals(0));
    });

    test('free', () {
      final docs = Documents(documents);
      expect(docs.ptr, isNot(equals(nullptr)));
      docs.free();
      expect(docs.ptr, equals(nullptr));
    });
  });

  group('Ranks', () {
    test('to list', () {
      final size = documents.length;
      final ranksPtr = malloc.call<Uint32>(size);
      final ranks = Ranks(ranksPtr, size);
      expect(ranks.toList().length, equals(size));
      malloc.free(ranksPtr);
    });

    test('empty', () {
      final ranks = Ranks(nullptr, 0);
      expect(ranks.toList(), isEmpty);
    });
  });
}
