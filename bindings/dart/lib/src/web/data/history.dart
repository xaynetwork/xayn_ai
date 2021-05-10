@JS()
library history;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show Feedback, History, Relevance;

extension RelevanceInt on Relevance {
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

  /// Creates the relevance level from a discriminant.
  static Relevance fromInt(int idx) {
    switch (idx) {
      case 0:
        return Relevance.low;
      case 1:
        return Relevance.medium;
      case 2:
        return Relevance.high;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

extension FeedbackInt on Feedback {
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

  /// Creates the feedback level from a discriminant.
  static Feedback fromInt(int idx) {
    switch (idx) {
      case 0:
        return Feedback.relevant;
      case 1:
        return Feedback.irrelevant;
      case 2:
        return Feedback.notGiven;
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
