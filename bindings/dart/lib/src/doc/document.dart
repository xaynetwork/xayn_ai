import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart'
    show CFeedback, CRelevance;

/// A document relevance level.
enum Relevance {
  low,
  medium,
  high,
}

extension RelevanceInt on Relevance {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Relevance.low:
        return CRelevance.Low;
      case Relevance.medium:
        return CRelevance.Medium;
      case Relevance.high:
        return CRelevance.High;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }

  /// Creates the relevance level from a discriminant.
  static Relevance fromInt(int idx) {
    switch (idx) {
      case CRelevance.Low:
        return Relevance.low;
      case CRelevance.Medium:
        return Relevance.medium;
      case CRelevance.High:
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
  none,
}

extension FeedbackInt on Feedback {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Feedback.relevant:
        return CFeedback.Relevant;
      case Feedback.irrelevant:
        return CFeedback.Irrelevant;
      case Feedback.none:
        return CFeedback.None;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }

  /// Creates the feedback level from a discriminant.
  static Feedback fromInt(int idx) {
    switch (idx) {
      case CFeedback.Relevant:
        return Feedback.relevant;
      case CFeedback.Irrelevant:
        return Feedback.irrelevant;
      case CFeedback.None:
        return Feedback.none;
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
