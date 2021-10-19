import 'package:json_annotation/json_annotation.dart';
import 'package:xayn_ai_ffi_dart/src/web/worker/message.dart' show ToJson;

part 'outcomes.g.dart';

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
@JsonSerializable()
class RerankingOutcomes implements ToJson {
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

  RerankingOutcomes(
      this.finalRanks, this.qaMBertSimilarities, this.contextScores);

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

  factory RerankingOutcomes.fromJson(Map json) =>
      _$RerankingOutcomesFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$RerankingOutcomesToJson(this);
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
