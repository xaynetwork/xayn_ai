import 'package:json_annotation/json_annotation.dart' show JsonSerializable;
import 'package:meta/meta.dart' show immutable;

part 'analytics.g.dart';

/// The analytics of the Xayn AI.
@immutable
@JsonSerializable()
class Analytics {
  /// The nDCG@k score between the LTR ranking and the relevance based ranking.
  final double ndcgLtr;

  /// The nDCG@k score between the Context ranking and the relevance based ranking.
  final double ndcgContext;

  /// The nDCG@k score between the initial ranking and the relevance based ranking.
  final double ndcgInitialRanking;

  /// The nDCG@k score between the final ranking and the relevance based ranking.
  final double ndcgFinalRanking;

  /// Creates the analytics from the individual values.
  Analytics(
    this.ndcgLtr,
    this.ndcgContext,
    this.ndcgInitialRanking,
    this.ndcgFinalRanking,
  );

  factory Analytics.fromJson(Map json) => _$AnalyticsFromJson(json);

  Map<String, dynamic> toJson() => _$AnalyticsToJson(this);
}
