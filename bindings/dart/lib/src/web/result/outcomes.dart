@JS()
library outcomes;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;

@JS()
@anonymous
class JsRerankingOutcomes {
  // ignore: non_constant_identifier_names
  external List<int> final_ranking;

  // ignore: non_constant_identifier_names
  external List<double>? qa_mbert_similarities;

  // ignore: non_constant_identifier_names
  external List<double>? context_scores;
}

extension ToRerankingOutcomes on JsRerankingOutcomes {
  /// Creates reranking outcomes from the current JS representation.
  RerankingOutcomes toRerankingOutcomes() => RerankingOutcomes.fromParts(
        final_ranking,
        qa_mbert_similarities,
        context_scores,
      );
}
