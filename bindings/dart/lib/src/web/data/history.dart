@JS()
library history;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show
        FeedbackToInt,
        History,
        RelevanceToInt,
        DayOfWeekToInt,
        UserActionToInt;

@JS()
@anonymous
class JsHistory {
  external factory JsHistory({
    String id,
    int relevance,
    int feedback,
    String session,
    // ignore: non_constant_identifier_names
    int query_count,
    // ignore: non_constant_identifier_names
    String query_id,
    // ignore: non_constant_identifier_names
    String query_words,
    int day,
    String url,
    String domain,
    int rank,
    // ignore: non_constant_identifier_names
    int user_action,
  });
}

extension ToJsHistories on List<History> {
  /// Creates JS histories from the current histories.
  List<JsHistory> toJsHistories() => List.generate(
        length,
        (i) => JsHistory(
          id: this[i].id,
          relevance: this[i].relevance.toInt(),
          feedback: this[i].feedback.toInt(),
          session: this[i].session,
          query_count: this[i].queryCount,
          query_id: this[i].queryId,
          query_words: this[i].queryWords,
          day: this[i].day.toInt(),
          url: this[i].url,
          domain: this[i].domain,
          rank: this[i].rank.toInt(),
          user_action: this[i].userAction.toInt(),
        ),
        growable: false,
      );
}
