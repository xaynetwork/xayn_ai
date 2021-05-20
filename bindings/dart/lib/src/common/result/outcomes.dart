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
  final List<int> finalRanks;

  /// The QA-mBERT similarities in order of the input documents.
  ///
  /// Can be empty if not calculated.
  final List<double>? qaMBertSimilarities;

  /// The context scores for all documents in order of the input documents.
  ///
  /// Can be empty if not calculated.
  final List<double>? contextScores;

  /// Create a new instance from its parts.
  ///
  /// Besides for testing this should ONLY be used by the `mobile/` and `web/`
  /// FFI binding.
  ///
  RerankingOutcomes.fromParts(
    this.finalRanks,
    this.contextScores,
    this.qaMBertSimilarities,
  ) {
    checkOutcomeLength(contextScores, finalRanks, 'contextScores');
    checkOutcomeLength(qaMBertSimilarities, finalRanks, 'qaMBertSimilarities');
  }
}

void checkOutcomeLength<T, E>(List<T>? outcome, List<E> base, String name) {
  if (outcome == null) {
    return;
  }
  final outLen = outcome.length;
  final baseLen = base.length;
  if (outLen != baseLen) {
    throw ArgumentError(
        'Invalid Outcome length for $name: len=$outLen expected 0 or $baseLen');
  }
}
