import 'dart:ffi' show nullptr, AllocatorAlloc, StructPointer;

import 'package:ffi/ffi.dart' show calloc;
import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNotNull, isNull, test, throwsStateError;
import 'package:xayn_ai_ffi_dart/src/reranker/analytics.dart'
    show AnalyticsBuilder;
import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CAnalytics;

void main() {
  group('Analytics', () {
    test('create from C', () {
      final cAnalytics = calloc.call<CAnalytics>();
      cAnalytics.ref.ndcg_ltr = 0.25;
      cAnalytics.ref.ndcg_context = 0.75;
      cAnalytics.ref.ndcg_initial_ranking = 0.125;
      cAnalytics.ref.ndcg_final_ranking = -25.25;
      try {
        final builder = AnalyticsBuilder(cAnalytics);
        final analytics = builder.build();
        //satify dart non null analytics
        if (analytics == null) return expect(analytics, isNotNull);
        expect(analytics.ndcgLtr, equals(0.25));
        expect(analytics.ndcgContext, equals(0.75));
        expect(analytics.ndcgInitialRanking, equals(0.125));
        expect(analytics.ndcgFinalRanking, equals(-25.25));
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
