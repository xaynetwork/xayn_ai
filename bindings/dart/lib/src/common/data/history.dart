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
  final String id;
  final Relevance relevance;
  final Feedback feedback;

  /// Creates the document history.
  History(this.id, this.relevance, this.feedback) {
    if (id.isEmpty) {
      throw ArgumentError('empty document history id');
    }
  }
}
