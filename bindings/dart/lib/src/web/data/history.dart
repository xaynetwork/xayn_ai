import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show Feedback, Relevance;

extension RelevanceStr on Relevance {
  /// Gets the variant name.
  String toStr() {
    switch (this) {
      case Relevance.low:
        return 'Low';
      case Relevance.medium:
        return 'Medium';
      case Relevance.high:
        return 'High';
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }

  /// Creates the relevance level from a variant name.
  static Relevance fromStr(String variant) {
    switch (variant) {
      case 'Low':
        return Relevance.low;
      case 'Medium':
        return Relevance.medium;
      case 'High':
        return Relevance.high;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

extension FeedbackStr on Feedback {
  /// Gets the variant name.
  String toStr() {
    switch (this) {
      case Feedback.relevant:
        return 'Relevant';
      case Feedback.irrelevant:
        return 'Irrelevant';
      case Feedback.notGiven:
        return 'NotGiven';
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }

  /// Creates the feedback level from a variant name.
  static Feedback fromStr(String variant) {
    switch (variant) {
      case 'Relevant':
        return Feedback.relevant;
      case 'Irrelevant':
        return Feedback.irrelevant;
      case 'NotGiven':
        return Feedback.notGiven;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}
