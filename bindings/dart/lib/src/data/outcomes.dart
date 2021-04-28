import 'dart:ffi' show nullptr, Pointer, StructPointer;
import 'dart:typed_data' show Float32List, Uint16List;

import 'package:xayn_ai_ffi_dart/src/data/slice.dart'
    show BoxedSliceF32List, BoxedSliceU16List;
import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CRerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

/// Type containing all reranking outcomes.
///
/// Some of the outcomes can be empty if they
/// had not been calculated. This can happen due to
/// configurations when running rerank or if an
/// non-panic error happened during execution and
/// as such only partial results are available.
///
/// Note that `finalRanks` is empty if and only if there
/// had been no input documents.
class RerankingOutcomes {
  /// The final ranking in order of the input documents.
  ///
  /// Should only be empty if there where no input documents.
  final Uint16List finalRanks;

  /// The QA-mBERT similarities in order of the input documents.
  ///
  /// Can be empty if not calculated.
  final Float32List qaMBertSimilarities;

  /// The context scores for all documents in order of the input documents.
  ///
  /// Can be empty if not calculated.
  final Float32List contextScores;

  RerankingOutcomes._(
    this.finalRanks,
    this.contextScores,
    this.qaMBertSimilarities,
  );
}

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
      final contextScores = _cOutcomes.ref.context_scores.toList();
      final qaMBertSimilarities = _cOutcomes.ref.qa_mbert_similarities.toList();

      return RerankingOutcomes._(
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
