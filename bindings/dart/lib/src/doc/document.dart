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
  none,
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

  /// Gets the feedback.
  Feedback get feedback => _feedback;

  /// Gets the relevance.
  Relevance get relevance => _relevance;
}

/// The document.
class Document {
  final String _id;
  final String _snippet;
  final int _rank;

  /// Creates the document.
  Document(this._id, this._snippet, this._rank) {
    if (_id.isEmpty) {
      throw ArgumentError('empty document id');
    }
    if (_snippet.isEmpty) {
      throw ArgumentError('empty document snippet id');
    }
    if (_rank.isNegative) {
      throw ArgumentError('negative document rank');
    }
  }

  /// Gets the id.
  String get id => _id;

  /// Gets the snippet.
  String get snippet => _snippet;

  /// Gets the rank.
  int get rank => _rank;
}
