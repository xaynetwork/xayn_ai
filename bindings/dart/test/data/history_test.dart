import 'dart:ffi' show nullptr, StructPointer;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/src/data/history.dart'
    show Feedback, FeedbackCast, Histories, History, Relevance, RelevanceCast;
import '../utils.dart' show histories;

void main() {
  group('History', () {
    test('empty', () {
      expect(
        () => History('', Relevance.low, Feedback.irrelevant),
        throwsArgumentError,
      );
    });
  });

  group('Histories', () {
    test('new', () {
      final hists = Histories(histories);
      histories.asMap().forEach((i, history) {
        expect(
          hists.ptr.ref.data[i].id.cast<Utf8>().toDartString(),
          equals(history.id),
        );
        expect(
          hists.ptr.ref.data[i].relevance,
          equals(history.relevance.toInt()),
        );
        expect(
          hists.ptr.ref.data[i].feedback,
          equals(history.feedback.toInt()),
        );
      });
      expect(hists.ptr.ref.len, equals(histories.length));
      hists.free();
    });

    test('empty', () {
      final hists = Histories([]);
      expect(hists.ptr.ref.data, equals(nullptr));
      expect(hists.ptr.ref.len, equals(0));
      hists.free();
    });

    test('free', () {
      final hist = Histories(histories);
      expect(hist.ptr, isNot(equals(nullptr)));
      hist.free();
      expect(hist.ptr, equals(nullptr));
    });
  });
}
