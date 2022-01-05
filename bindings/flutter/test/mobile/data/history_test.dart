import 'dart:ffi' show nullptr, StructPointer;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNot, test, throwsArgumentError;

import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show
        DayOfWeek,
        DayOfWeekToInt,
        UserFeedback,
        FeedbackToInt,
        History,
        Relevance,
        RelevanceToInt,
        UserAction,
        UserActionToInt;
import 'package:xayn_ai_ffi_dart/src/mobile/data/history.dart' show Histories;
import '../utils.dart' show histories;

void main() {
  group('History', () {
    test('empty', () {
      final id = 'fcb6a685-eb92-4d36-8686-000000000000';
      expect(
        () => History(
          id: '', // empty id
          relevance: Relevance.low,
          userFeedback: UserFeedback.irrelevant,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          day: DayOfWeek.mon,
          url: 'url',
          domain: 'domain',
          rank: 0,
          userAction: UserAction.miss,
        ),
        throwsArgumentError,
      );
      expect(
        () => History(
          id: id,
          relevance: Relevance.low,
          userFeedback: UserFeedback.irrelevant,
          session: '', // empty session id
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          day: DayOfWeek.mon,
          url: 'url',
          domain: 'domain',
          rank: 0,
          userAction: UserAction.miss,
        ),
        throwsArgumentError,
      );
      expect(
        () => History(
          id: id,
          relevance: Relevance.low,
          userFeedback: UserFeedback.irrelevant,
          session: id,
          queryCount: -10, // negative query count
          queryId: id,
          queryWords: 'query words',
          day: DayOfWeek.mon,
          url: 'url',
          domain: 'domain',
          rank: 0,
          userAction: UserAction.miss,
        ),
        throwsArgumentError,
      );
      expect(
        () => History(
          id: id,
          relevance: Relevance.low,
          userFeedback: UserFeedback.irrelevant,
          session: id,
          queryCount: 1,
          queryId: '', // empty query id
          queryWords: 'query words',
          day: DayOfWeek.mon,
          url: 'url',
          domain: 'domain',
          rank: 0,
          userAction: UserAction.miss,
        ),
        throwsArgumentError,
      );
      expect(
        () => History(
          id: id,
          relevance: Relevance.low,
          userFeedback: UserFeedback.irrelevant,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: '', // empty query words
          day: DayOfWeek.mon,
          url: 'url',
          domain: 'domain',
          rank: 0,
          userAction: UserAction.miss,
        ),
        throwsArgumentError,
      );
      expect(
        () => History(
          id: id,
          relevance: Relevance.low,
          userFeedback: UserFeedback.irrelevant,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          day: DayOfWeek.mon,
          url: '', // empty url
          domain: 'domain',
          rank: 0,
          userAction: UserAction.miss,
        ),
        throwsArgumentError,
      );
      expect(
        () => History(
          id: id,
          relevance: Relevance.low,
          userFeedback: UserFeedback.irrelevant,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          day: DayOfWeek.mon,
          url: 'url',
          domain: '', // empty domain
          rank: 0,
          userAction: UserAction.miss,
        ),
        throwsArgumentError,
      );
      expect(
        () => History(
          id: id,
          relevance: Relevance.low,
          userFeedback: UserFeedback.irrelevant,
          session: id,
          queryCount: 1,
          queryId: id,
          queryWords: 'query words',
          day: DayOfWeek.mon,
          url: 'url',
          domain: 'domain',
          rank: -1, // negative rank
          userAction: UserAction.miss,
        ),
        throwsArgumentError,
      );
    });
  });

  group('Histories', () {
    test('new', () {
      final hists = Histories(histories);
      histories.asMap().forEach((i, history) {
        var hist = hists.ptr.ref.data[i];
        expect(
          hist.id.cast<Utf8>().toDartString(),
          equals(history.id),
        );
        expect(
          hist.relevance,
          equals(history.relevance.toInt()),
        );
        expect(
          hist.user_feedback,
          equals(history.userFeedback.toInt()),
        );
        expect(
          hist.session.cast<Utf8>().toDartString(),
          equals(history.session),
        );
        expect(
          hist.query_count,
          equals(history.queryCount),
        );
        expect(
          hist.query_id.cast<Utf8>().toDartString(),
          equals(history.queryId),
        );
        expect(
          hist.query_words.cast<Utf8>().toDartString(),
          equals(history.queryWords),
        );
        expect(
          hist.day,
          equals(history.day.toInt()),
        );
        expect(
          hist.url.cast<Utf8>().toDartString(),
          equals(history.url),
        );
        expect(
          hist.domain.cast<Utf8>().toDartString(),
          equals(history.domain),
        );
        expect(
          hist.rank,
          equals(history.rank),
        );
        expect(
          hist.user_action,
          equals(history.userAction.toInt()),
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
