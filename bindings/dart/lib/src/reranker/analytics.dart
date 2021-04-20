import 'dart:ffi' show nullptr, Pointer;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CAnalytics;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

/// The analytics of the penultimate reranking.
class Analytics {
  late Pointer<CAnalytics> _analytics;

  /// Creates the analytics.
  ///
  /// This constructor never throws an exception.
  Analytics(this._analytics);

  /// Frees the memory.
  void free() {
    if (_analytics != nullptr) {
      ffi.analytics_drop(_analytics);
      _analytics = nullptr;
    }
  }
}
