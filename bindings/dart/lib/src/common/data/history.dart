import 'package:meta/meta.dart' show immutable;

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

/// Day of week.
enum DayOfWeek {
  mon,
  tue,
  wed,
  thu,
  fri,
  sat,
  sun,
}

/// A user interaction.
enum UserAction {
  miss,
  skip,
  click,
}

/// The document history.
@immutable
class History {
  final String id;
  final Relevance relevance;
  final Feedback feedback;
  final String session;
  final int queryCount;
  final String queryId;
  final String queryWords;
  final DayOfWeek day;
  final String url;
  final String domain;
  final int rank;
  final UserAction userAction;

  /// Creates the document history.
  History(
      this.id,
      this.relevance,
      this.feedback,
      this.session,
      this.queryCount,
      this.queryId,
      this.queryWords,
      this.day,
      this.url,
      this.domain,
      this.rank,
      this.userAction) {
    if (id.isEmpty) {
      throw ArgumentError('empty document history id');
    }
    if (session.isEmpty) {
      throw ArgumentError('empty session id');
    }
    if (queryCount < 1) {
      throw ArgumentError('non-positive query count');
    }
    if (queryId.isEmpty) {
      throw ArgumentError('empty query id');
    }
    if (queryWords.isEmpty) {
      throw ArgumentError('empty query words');
    }
    if (url.isEmpty) {
      throw ArgumentError('empty document url');
    }
    if (domain.isEmpty) {
      throw ArgumentError('empty document domain');
    }
    if (rank.isNegative) {
      throw ArgumentError('negative document rank');
    }
  }
}
