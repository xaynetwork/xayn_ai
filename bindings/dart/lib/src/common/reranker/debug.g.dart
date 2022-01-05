// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'debug.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

RerankDebugCallData _$RerankDebugCallDataFromJson(Map json) =>
    RerankDebugCallData(
      rerankMode: _$enumDecode(_$RerankModeEnumMap, json['rerank_mode']),
      histories: (json['histories'] as List<dynamic>)
          .map((e) => History.fromJson(Map<String, dynamic>.from(e as Map)))
          .toList(),
      documents: (json['documents'] as List<dynamic>)
          .map((e) => Document.fromJson(Map<String, dynamic>.from(e as Map)))
          .toList(),
      serializedState: _optBase64ToBytes(json['serialized_state'] as String?),
    );

Map<String, dynamic> _$RerankDebugCallDataToJson(
        RerankDebugCallData instance) =>
    <String, dynamic>{
      'rerank_mode': _$RerankModeEnumMap[instance.rerankMode],
      'histories': instance.histories.map((e) => e.toJson()).toList(),
      'documents': instance.documents.map((e) => e.toJson()).toList(),
      'serialized_state': _optBytesToBase64(instance.serializedState),
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

const _$RerankModeEnumMap = {
  RerankMode.standardNews: 0,
  RerankMode.personalizedNews: 1,
  RerankMode.standardSearch: 2,
  RerankMode.personalizedSearch: 3,
};
