// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'history.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

History _$HistoryFromJson(Map json) => History(
      id: json['id'] as String,
      relevance: _$enumDecode(_$RelevanceEnumMap, json['relevance']),
      userFeedback: _$enumDecode(_$UserFeedbackEnumMap, json['user_feedback']),
      session: json['session'] as String,
      queryCount: json['query_count'] as int,
      queryId: json['query_id'] as String,
      queryWords: json['query_words'] as String,
      day: _$enumDecode(_$DayOfWeekEnumMap, json['day']),
      url: json['url'] as String,
      domain: json['domain'] as String,
      rank: json['rank'] as int,
      userAction: _$enumDecode(_$UserActionEnumMap, json['user_action']),
    );

Map<String, dynamic> _$HistoryToJson(History instance) => <String, dynamic>{
      'id': instance.id,
      'relevance': _$RelevanceEnumMap[instance.relevance],
      'user_feedback': _$UserFeedbackEnumMap[instance.userFeedback],
      'session': instance.session,
      'query_count': instance.queryCount,
      'query_id': instance.queryId,
      'query_words': instance.queryWords,
      'day': _$DayOfWeekEnumMap[instance.day],
      'url': instance.url,
      'domain': instance.domain,
      'rank': instance.rank,
      'user_action': _$UserActionEnumMap[instance.userAction],
    };

K _$enumDecode<K, V>(
  Map<K, V> enumValues,
  Object? source, {
  K? unknownValue,
}) {
  if (source == null) {
    throw ArgumentError(
      'A value must be provided. Supported values: '
      '${enumValues.values.join(', ')}',
    );
  }

  return enumValues.entries.singleWhere(
    (e) => e.value == source,
    orElse: () {
      if (unknownValue == null) {
        throw ArgumentError(
          '`$source` is not one of the supported values: '
          '${enumValues.values.join(', ')}',
        );
      }
      return MapEntry(unknownValue, enumValues.values.first);
    },
  ).key;
}

const _$RelevanceEnumMap = {
  Relevance.low: 0,
  Relevance.medium: 1,
  Relevance.high: 2,
};

const _$UserFeedbackEnumMap = {
  UserFeedback.relevant: 0,
  UserFeedback.irrelevant: 1,
  UserFeedback.notGiven: 2,
};

const _$DayOfWeekEnumMap = {
  DayOfWeek.mon: 0,
  DayOfWeek.tue: 1,
  DayOfWeek.wed: 2,
  DayOfWeek.thu: 3,
  DayOfWeek.fri: 4,
  DayOfWeek.sat: 5,
  DayOfWeek.sun: 6,
};

const _$UserActionEnumMap = {
  UserAction.miss: 0,
  UserAction.skip: 1,
  UserAction.click: 2,
};
