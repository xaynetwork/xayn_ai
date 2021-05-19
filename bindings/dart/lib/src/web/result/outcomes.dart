@JS()
library outcomes;

import 'dart:typed_data' show Float32List, Uint16List;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;

@JS()
@anonymous
class JsRerankingOutcomes {
  // ignore: non_constant_identifier_names
  external Uint16List final_ranking;

  // ignore: non_constant_identifier_names
  external Float32List? qa_mbert_similarities;

  // ignore: non_constant_identifier_names
  external Float32List? context_scores;
}

extension ToRerankingOutcomes on JsRerankingOutcomes {
  /// Creates reranking outcomes from the current JS representation.
  RerankingOutcomes toRerankingOutcomes() => RerankingOutcomes.fromParts(
        final_ranking,
        qa_mbert_similarities,
        context_scores,
      );
}
