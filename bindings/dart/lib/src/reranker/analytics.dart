import 'dart:ffi' show nullptr, Pointer, StructPointer;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CAnalytics;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

class Analytics {
  /// The nDCG@k score between the LTR ranking and the relevance based ranking
  final double ndcgLtr;

  /// The nDCG@k score between the Context ranking and the relevance based ranking
  final double ndcgContext;

  /// The nDCG@k score between the initial ranking and the relevance based ranking
  final double ndcgInitialRanking;

  /// The nDCG@k score between the final ranking and the relevance based ranking
  final double ndcgFinalRanking;

  Analytics._(this.ndcgLtr, this.ndcgContext, this.ndcgInitialRanking,
      this.ndcgFinalRanking);

  static AnalyticsFromCBuilder fromCBuilder(Pointer<CAnalytics> cAnalytics) =>
      AnalyticsFromCBuilder._(cAnalytics);
}

class AnalyticsFromCBuilder {
  final Pointer<CAnalytics> _cAnalytics;
  bool _freed = false;

  AnalyticsFromCBuilder._(this._cAnalytics);

  Analytics? buildFromC() {
    if (_freed) {
      throw StateError('CAnalytics already freed');
    } else if (_cAnalytics == nullptr) {
      // Analytics might not be provided
      return null;
    } else {
      final cval = _cAnalytics.ref;
      return Analytics._(cval.ndcg_ltr, cval.ndcg_context,
          cval.ndcg_initial_ranking, cval.ndcg_final_ranking);
    }
  }

  /// Frees the memory.
  void free() {
    if (!_freed) {
      _freed = true;
      // drop impl's are nullptr safe, but we don't want to call into ffi in tests
      if (_cAnalytics != nullptr) ffi.analytics_drop(_cAnalytics);
    } else {
      throw StateError('CAnalytics already freed (double free)');
    }
  }
}
