/// A document relevance level.
enum Relevance {
  low,
  medium,
  high,
}

/// A user feedback level.
enum Feedback {
  relevant,
  irrelevant,
  notGiven,
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
