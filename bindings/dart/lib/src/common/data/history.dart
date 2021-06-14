import 'package:meta/meta.dart' show immutable;

import 'package:xayn_ai_ffi_dart/src/common/ffi/genesis.dart' as ffi;

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
        return ffi.Relevance.Low;
      case Relevance.medium:
        return ffi.Relevance.Medium;
      case Relevance.high:
        return ffi.Relevance.High;
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
        return ffi.UserFeedback.Relevant;
      case Feedback.irrelevant:
        return ffi.UserFeedback.Irrelevant;
      case Feedback.notGiven:
        return ffi.UserFeedback.NotGiven;
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
        return ffi.DayOfWeek.Mon;
      case DayOfWeek.tue:
        return ffi.DayOfWeek.Tue;
      case DayOfWeek.wed:
        return ffi.DayOfWeek.Wed;
      case DayOfWeek.thu:
        return ffi.DayOfWeek.Thu;
      case DayOfWeek.fri:
        return ffi.DayOfWeek.Fri;
      case DayOfWeek.sat:
        return ffi.DayOfWeek.Sat;
      case DayOfWeek.sun:
        return ffi.DayOfWeek.Sun;
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
        return ffi.UserAction.Miss;
      case UserAction.skip:
        return ffi.UserAction.Skip;
      case UserAction.click:
        return ffi.UserAction.Click;
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
