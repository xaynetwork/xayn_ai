// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'response.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Response _$ResponseFromJson(Map json) => Response(
      _$enumDecode(_$ResultEnumMap, json['kind']),
      (json['result'] as Map?)?.map(
        (k, e) => MapEntry(k as String, e),
      ),
    );

Map<String, dynamic> _$ResponseToJson(Response instance) => <String, dynamic>{
      'kind': _$ResultEnumMap[instance.kind],
      'result': instance.result,
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

const _$ResultEnumMap = {
  Result.ok: 'ok',
  Result.exception: 'exception',
};

Uint8ListResponse _$Uint8ListResponseFromJson(Map json) => Uint8ListResponse(
      const Uint8ListConverter().fromJson(json['data'] as Uint8List),
    );

Map<String, dynamic> _$Uint8ListResponseToJson(Uint8ListResponse instance) =>
    <String, dynamic>{
      'data': const Uint8ListConverter().toJson(instance.data),
    };

FaultsResponse _$FaultsResponseFromJson(Map json) => FaultsResponse(
      (json['faults'] as List<dynamic>).map((e) => e as String).toList(),
    );

Map<String, dynamic> _$FaultsResponseToJson(FaultsResponse instance) =>
    <String, dynamic>{
      'faults': instance.faults,
    };

AnalyticsResponse _$AnalyticsResponseFromJson(Map json) => AnalyticsResponse(
      json['analytics'] == null
          ? null
          : Analytics.fromJson(json['analytics'] as Map),
    );

Map<String, dynamic> _$AnalyticsResponseToJson(AnalyticsResponse instance) =>
    <String, dynamic>{
      'analytics': instance.analytics?.toJson(),
    };
