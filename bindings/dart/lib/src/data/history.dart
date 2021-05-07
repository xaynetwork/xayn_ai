/// A document relevance level.
enum Relevance {
  low,
  medium,
  high,
}

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

/// A user feedback level.
enum Feedback {
  relevant,
  irrelevant,
  notGiven,
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

/// The document history.
class History {
  final String _id;
  final Relevance _relevance;
  final Feedback _feedback;

  /// Creates the document history.
  History(this._id, this._relevance, this._feedback) {
    if (_id.isEmpty) {
      throw ArgumentError('empty document history id');
    }
  }

  /// Gets the id.
  String get id => _id;

  /// Gets the relevance level.
  Relevance get relevance => _relevance;

  /// Gets the user feedback.
  Feedback get feedback => _feedback;
}
