@JS()
library analytics;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;

@JS()
@anonymous
class JsAnalytics {
  // ignore: non_constant_identifier_names
  external double get ndcg_ltr;
  // ignore: non_constant_identifier_names
  external double get ndcg_context;
  // ignore: non_constant_identifier_names
  external double get ndcg_initial_ranking;
  // ignore: non_constant_identifier_names
  external double get ndcg_final_ranking;
  external factory JsAnalytics({
    // ignore: non_constant_identifier_names, unused_element
    double ndcg_ltr,
    // ignore: non_constant_identifier_names, unused_element
    double ndcg_context,
    // ignore: non_constant_identifier_names, unused_element
    double ndcg_initial_ranking,
    // ignore: non_constant_identifier_names, unused_element
    double ndcg_final_ranking,
  });
}

extension ToAnalytics on JsAnalytics {
  Analytics toAnalytics() => Analytics(
        ndcg_ltr,
        ndcg_context,
        ndcg_initial_ranking,
        ndcg_final_ranking,
      );
}
