// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'outcomes.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

RerankingOutcomes _$RerankingOutcomesFromJson(Map json) => RerankingOutcomes(
      (json['final_ranks'] as List<dynamic>).map((e) => e as int).toList(),
      (json['qa_m_bert_similarities'] as List<dynamic>?)
          ?.map((e) => (e as num).toDouble())
          .toList(),
      (json['context_scores'] as List<dynamic>?)
          ?.map((e) => (e as num).toDouble())
          .toList(),
    );

Map<String, dynamic> _$RerankingOutcomesToJson(RerankingOutcomes instance) =>
    <String, dynamic>{
      'final_ranks': instance.finalRanks,
      'qa_m_bert_similarities': instance.qaMBertSimilarities,
      'context_scores': instance.contextScores,
    };
