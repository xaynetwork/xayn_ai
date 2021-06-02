@JS()
library history;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show Feedback, History, Relevance, DayOfWeek, UserAction;

extension RelevanceToInt on Relevance {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Relevance.low:
        return 0;
      case Relevance.medium:
        return 1;
      case Relevance.high:
        return 2;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

extension FeedbackToInt on Feedback {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Feedback.relevant:
        return 0;
      case Feedback.irrelevant:
        return 1;
      case Feedback.notGiven:
        return 2;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

extension DayOfWeekToInt on DayOfWeek {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case DayOfWeek.mon:
        return 0;
      case DayOfWeek.tue:
        return 1;
      case DayOfWeek.wed:
        return 2;
      case DayOfWeek.thu:
        return 3;
      case DayOfWeek.fri:
        return 4;
      case DayOfWeek.sat:
        return 5;
      case DayOfWeek.sun:
        return 6;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

extension UserActionToInt on UserAction {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case UserAction.miss:
        return 0;
      case UserAction.skip:
        return 1;
      case UserAction.click:
        return 2;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

@JS()
@anonymous
class JsHistory {
  external factory JsHistory(
      {String id,
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
      int user_action});
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
