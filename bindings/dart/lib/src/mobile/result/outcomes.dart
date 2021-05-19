import 'dart:ffi' show nullptr, Pointer, StructPointer;

import 'package:xayn_ai_ffi_dart/src/mobile/result/slice.dart'
    show BoxedSliceF32List, BoxedSliceU16List;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CRerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/library.dart' show ffi;

class RerankingOutcomesBuilder {
  final Pointer<CRerankingOutcomes> _cOutcomes;
  bool _freed = false;

  RerankingOutcomesBuilder(this._cOutcomes);

  /// Build the `RerankingOutcomes` based on the passed in pointer.
  ///
  /// This should be called in a `try {} finally {}` block where in
  /// the `finally` block `builder.free()` is called.
  ///
  /// If this is called and the pointer with which this instance was
  /// created was a `nullptr` a exception is thrown. As you *should* only
  /// call this after checking the error codes this should not happen in
  /// practice.
  RerankingOutcomes build() {
    if (_freed) {
      throw StateError('CRerankingOutcomes have already been freed');
    } else if (_cOutcomes == nullptr) {
      throw StateError(
          'Error codes should be checked befor building outcomes from C.');
    } else {
      final finalRanking = _cOutcomes.ref.final_ranking.toList();
      if (finalRanking == null) {
        throw ArgumentError('Final rankings outcome was null.');
      }
      final contextScores = _cOutcomes.ref.context_scores.toList();
      final qaMBertSimilarities = _cOutcomes.ref.qa_mbert_similarities.toList();

      return RerankingOutcomes.fromParts(
        finalRanking,
        contextScores,
        qaMBertSimilarities,
      );
    }
  }

  /// Free the wrapped C struct by calling the rust FFI.
  void free() {
    if (!_freed) {
      _freed = true;
      // can always be called with a nullptr, but in test we don't want to call ffi
      if (_cOutcomes != nullptr) ffi.reranking_outcomes_drop(_cOutcomes);
    }
  }
}
