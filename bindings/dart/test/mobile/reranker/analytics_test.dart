import 'dart:ffi' show nullptr, AllocatorAlloc, StructPointer;

import 'package:ffi/ffi.dart' show calloc;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNotNull, isNull, test, throwsStateError;
import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show History, Relevance, UserAction, UserFeedback, DayOfWeek;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart' show CAnalytics;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/analytics.dart'
    show AnalyticsBuilder;

import '../utils.dart' show mkSetupData;

void main() {
  group('Analytics', () {
    test('AI return analytics', () async {
      final ai = await XaynAi.create(mkSetupData());
      final documents = [
        Document(
          id: 'fcb6a685-eb92-4d36-8686-8a70a3a33003',
          title: 'a b c',
          snippet: 'snippet of a b c',
          rank: 0,
          session: 'fcb6a685-eb92-4d36-8686-000000000100',
          queryCount: 21,
          queryId: 'fcb6a685-eb92-4d36-8686-A000A00B000B',
          queryWords: 'abc',
          url: 'url',
          domain: 'dom',
        ),
        Document(
          id: 'fcb6a685-eb92-4d36-8686-8a70a3a33004',
          title: 'ab de',
          snippet: 'snippet of ab de',
          rank: 1,
          session: 'fcb6a685-eb92-4d36-8686-000000000100',
          queryCount: 21,
          queryId: 'fcb6a685-eb92-4d36-8686-A000A00B000B',
          queryWords: 'abc',
          url: 'url2',
          domain: 'dom2',
        ),
      ];

      final histories = [
        History(
          id: 'fcb6a685-eb92-4d36-8686-8a70a3a33003',
          relevance: Relevance.high,
          userFeedback: UserFeedback.relevant,
          session: 'fcb6a685-eb92-4d36-8686-000000000100',
          queryCount: 1,
          queryId: 'fcb6a685-eb92-4d36-8686-A000A00A000A',
          queryWords: 'is the dodo alive',
          day: DayOfWeek.sun,
          url: 'dodo lives:or not',
          domain: 'no domain',
          rank: 0,
          userAction: UserAction.click,
        ),
        History(
          id: 'fcb6a685-eb92-4d36-8686-8a70a3a33004',
          relevance: Relevance.high,
          userFeedback: UserFeedback.relevant,
          session: 'fcb6a685-eb92-4d36-8686-000000000100',
          queryCount: 1,
          queryId: 'fcb6a685-eb92-4d36-8686-A000A00A000A',
          queryWords: 'is the dodo alive',
          day: DayOfWeek.sun,
          url: 'dodo lives:or not',
          domain: 'no domain',
          rank: 1,
          userAction: UserAction.click,
        ),
      ];

      // first rerank will return same order as api
      await ai.rerank(RerankMode.personalizedSearch, <History>[], documents);
      // second rerank create the coi in the feedbackloop
      // and it will be able to rerank properly
      await ai.rerank(RerankMode.personalizedSearch, histories, documents);
      // here we don't have analytics because the previous rerank
      // returned the rank from the api
      expect(await ai.analytics(), isNull);

      await ai.rerank(RerankMode.personalizedSearch, histories, documents);
      // this are the analytics about the second rerank
      expect(await ai.analytics(), isNotNull);
    });

    test('create from C', () {
      final cAnalytics = calloc.call<CAnalytics>();
      cAnalytics.ref.ndcg_ltr = 0.25;
      cAnalytics.ref.ndcg_context = 0.75;
      cAnalytics.ref.ndcg_initial_ranking = 0.125;
      cAnalytics.ref.ndcg_final_ranking = -25.25;
      try {
        final builder = AnalyticsBuilder(cAnalytics);
        final analytics = builder.build();
        expect(analytics, isNotNull);
        expect(analytics?.ndcgLtr, equals(0.25));
        expect(analytics?.ndcgContext, equals(0.75));
        expect(analytics?.ndcgInitialRanking, equals(0.125));
        expect(analytics?.ndcgFinalRanking, equals(-25.25));
        //must not call builder.free()
      } finally {
        calloc.free(cAnalytics);
      }
    });

    test('can be empty', () {
      final builder = AnalyticsBuilder(nullptr);
      final analytics = builder.build();
      expect(analytics, isNull);
    });

    test('throw error on use after free', () {
      final builder = AnalyticsBuilder(nullptr);
      builder.free();
      expect(() {
        builder.build();
      }, throwsStateError);
    });
  });
}
