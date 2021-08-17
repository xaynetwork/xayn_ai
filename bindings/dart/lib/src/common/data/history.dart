import 'package:json_annotation/json_annotation.dart'
    show JsonSerializable, JsonValue;
import 'package:meta/meta.dart' show immutable;
import 'package:xayn_ai_ffi_dart/src/common/ffi/genesis.dart' as ffi
    show Relevance, UserFeedback, DayOfWeek, UserAction;

part 'history.g.dart';

/// A document relevance level.
enum Relevance {
  @JsonValue(ffi.Relevance.Low)
  low,
  @JsonValue(ffi.Relevance.Medium)
  medium,
  @JsonValue(ffi.Relevance.High)
  high,
}

extension RelevanceToInt on Relevance {
  /// Gets the discriminant.
  int toInt() => _$RelevanceEnumMap[this]!;
}

/// A user feedback level.
enum UserFeedback {
  @JsonValue(ffi.UserFeedback.Relevant)
  relevant,
  @JsonValue(ffi.UserFeedback.Irrelevant)
  irrelevant,
  @JsonValue(ffi.UserFeedback.NotGiven)
  notGiven,
}

extension FeedbackToInt on UserFeedback {
  /// Gets the discriminant.
  int toInt() => _$UserFeedbackEnumMap[this]!;
}

/// Day of week.
enum DayOfWeek {
  @JsonValue(ffi.DayOfWeek.Mon)
  mon,
  @JsonValue(ffi.DayOfWeek.Tue)
  tue,
  @JsonValue(ffi.DayOfWeek.Wed)
  wed,
  @JsonValue(ffi.DayOfWeek.Thu)
  thu,
  @JsonValue(ffi.DayOfWeek.Fri)
  fri,
  @JsonValue(ffi.DayOfWeek.Sat)
  sat,
  @JsonValue(ffi.DayOfWeek.Sun)
  sun,
}

extension DayOfWeekToInt on DayOfWeek {
  /// Gets the discriminant.
  int toInt() => _$DayOfWeekEnumMap[this]!;
}

/// A user interaction.
enum UserAction {
  @JsonValue(ffi.UserAction.Miss)
  miss,
  @JsonValue(ffi.UserAction.Skip)
  skip,
  @JsonValue(ffi.UserAction.Click)
  click,
}

extension UserActionToInt on UserAction {
  /// Gets the discriminant.
  int toInt() => _$UserActionEnumMap[this]!;
}

/// The document history.
@JsonSerializable()
@immutable
class History {
  /// Unique identifier of the document.
  final String id;

  /// Relevance level of the document.
  final Relevance relevance;

  /// A flag that indicates whether the user liked the document.
  final UserFeedback userFeedback;

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
  History({
    required this.id,
    required this.relevance,
    required this.userFeedback,
    required this.session,
    required this.queryCount,
    required this.queryId,
    required this.queryWords,
    required this.day,
    required this.url,
    required this.domain,
    required this.rank,
    required this.userAction,
  }) {
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

  factory History.fromJson(Map<String, dynamic> json) =>
      _$HistoryFromJson(json);

  Map<String, dynamic> toJson() => _$HistoryToJson(this);
}
