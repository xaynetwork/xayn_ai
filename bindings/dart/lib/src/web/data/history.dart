@JS()
library history;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show Feedback, History, Relevance;

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

@JS()
@anonymous
class JsHistory {
  external factory JsHistory({String id, int relevance, int feedback});
}

extension ToJsHistories on List<History> {
  /// Creates JS histories from the current histories.
  List<JsHistory> toHistories() => List.generate(
        length,
        (i) => JsHistory(
          id: this[i].id,
          relevance: this[i].relevance.toInt(),
          feedback: this[i].feedback.toInt(),
        ),
        growable: false,
      );
}
