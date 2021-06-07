import 'package:meta/meta.dart' show immutable;

import 'package:xayn_ai_ffi_dart/src/common/ffi/genesis.dart'
    show CDayOfWeek, CFeedback, CRelevance, CUserAction;

/// A document relevance level.
enum Relevance {
  low,
  medium,
  high,
}

extension RelevanceToInt on Relevance {
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
}

/// A user feedback level.
enum Feedback {
  relevant,
  irrelevant,
  notGiven,
}

extension FeedbackToInt on Feedback {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Feedback.relevant:
        return CFeedback.Relevant;
      case Feedback.irrelevant:
        return CFeedback.Irrelevant;
      case Feedback.notGiven:
        return CFeedback.NotGiven;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
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

extension DayOfWeekToInt on DayOfWeek {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case DayOfWeek.mon:
        return CDayOfWeek.Mon;
      case DayOfWeek.tue:
        return CDayOfWeek.Tue;
      case DayOfWeek.wed:
        return CDayOfWeek.Wed;
      case DayOfWeek.thu:
        return CDayOfWeek.Thu;
      case DayOfWeek.fri:
        return CDayOfWeek.Fri;
      case DayOfWeek.sat:
        return CDayOfWeek.Sat;
      case DayOfWeek.sun:
        return CDayOfWeek.Sun;
    }
  }
}

/// A user interaction.
enum UserAction {
  miss,
  skip,
  click,
}

extension UserActionToInt on UserAction {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case UserAction.miss:
        return CUserAction.Miss;
      case UserAction.skip:
        return CUserAction.Skip;
      case UserAction.click:
        return CUserAction.Click;
    }
  }
}

/// The document history.
@immutable
class History {
  /// Unique identifier of the document.
  final String id;

  /// Relevance level of the document.
  final Relevance relevance;

  /// A flag that indicates whether the user liked the document.
  final Feedback feedback;

  /// Session of the document.
  final String session;

  /// Query count within session.
  final int queryCount;

  /// Query identifier of the document.
  final String queryId;

  /// Query of the document.
  final String queryWords;

  /// Day of week query was performed.
  final DayOfWeek day;

  /// URL of the document.
  final String url;

  /// Domain of the document.
  final String domain;

  /// Ranked position of the document.
  final int rank;

  /// User interaction for the document
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
