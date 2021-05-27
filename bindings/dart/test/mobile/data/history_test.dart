import 'dart:ffi' show nullptr, StructPointer;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show Feedback, Relevance;
import 'package:xayn_ai_ffi_dart/src/mobile/data/history.dart'
    show
        FeedbackToInt,
        Histories,
        RelevanceToInt,
        DayOfWeekToInt,
        UserActionToInt;
import '../utils.dart' show histories, mkTestHist;

void main() {
  group('History', () {
    test('empty', () {
      expect(
        () => mkTestHist('', Relevance.low, Feedback.irrelevant),
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
        expect(
          hists.ptr.ref.data[i].session.cast<Utf8>().toDartString(),
          equals(history.session),
        );
        expect(
          hists.ptr.ref.data[i].query_count,
          equals(history.queryCount),
        );
        expect(
          hists.ptr.ref.data[i].query_id.cast<Utf8>().toDartString(),
          equals(history.queryId),
        );
        expect(
          hists.ptr.ref.data[i].query_words.cast<Utf8>().toDartString(),
          equals(history.queryWords),
        );
        expect(
          hists.ptr.ref.data[i].day,
          equals(history.day.toInt()),
        );
        expect(
          hists.ptr.ref.data[i].url.cast<Utf8>().toDartString(),
          equals(history.url),
        );
        expect(
          hists.ptr.ref.data[i].domain.cast<Utf8>().toDartString(),
          equals(history.domain),
        );
        expect(
          hists.ptr.ref.data[i].rank,
          equals(history.rank),
        );
        expect(
          hists.ptr.ref.data[i].user_action,
          history.userAction.toInt(),
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
