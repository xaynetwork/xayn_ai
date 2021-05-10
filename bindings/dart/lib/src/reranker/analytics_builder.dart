import 'dart:ffi' show nullptr, Pointer, StructPointer;

import 'package:xayn_ai_ffi_dart/src/ffi/c/genesis.dart' show CAnalytics;
import 'package:xayn_ai_ffi_dart/src/ffi/c/library.dart' show ffi;
import 'package:xayn_ai_ffi_dart/src/reranker/analytics.dart' show Analytics;

class AnalyticsBuilder {
  final Pointer<CAnalytics> _cAnalytics;
  bool _freed = false;

  AnalyticsBuilder(this._cAnalytics);

  Analytics? build() {
    if (_freed) {
      throw StateError('CAnalytics already freed');
    } else if (_cAnalytics == nullptr) {
      // Analytics might not be provided
      return null;
    } else {
      final cval = _cAnalytics.ref;
      return Analytics(
        cval.ndcg_ltr,
        cval.ndcg_context,
        cval.ndcg_initial_ranking,
        cval.ndcg_final_ranking,
      );
    }
  }

  /// Frees the memory.
  void free() {
    if (!_freed) {
      _freed = true;
      // drop impl's are nullptr safe, but we don't want to call into ffi in tests
      if (_cAnalytics != nullptr) ffi.analytics_drop(_cAnalytics);
    }
  }
}
