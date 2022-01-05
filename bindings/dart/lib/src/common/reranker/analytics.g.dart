// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'analytics.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Analytics _$AnalyticsFromJson(Map json) => Analytics(
      (json['ndcg_ltr'] as num).toDouble(),
      (json['ndcg_context'] as num).toDouble(),
      (json['ndcg_initial_ranking'] as num).toDouble(),
      (json['ndcg_final_ranking'] as num).toDouble(),
    );

Map<String, dynamic> _$AnalyticsToJson(Analytics instance) => <String, dynamic>{
      'ndcg_ltr': instance.ndcgLtr,
      'ndcg_context': instance.ndcgContext,
      'ndcg_initial_ranking': instance.ndcgInitialRanking,
      'ndcg_final_ranking': instance.ndcgFinalRanking,
    };
