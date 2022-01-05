// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'request.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Request _$RequestFromJson(Map json) => Request(
      _$enumDecode(_$MethodEnumMap, json['method']),
      (json['params'] as Map?)?.map(
        (k, e) => MapEntry(k as String, e),
      ),
      Sender.fromJson(json['sender'] as Map),
    );

Map<String, dynamic> _$RequestToJson(Request instance) => <String, dynamic>{
      'method': _$MethodEnumMap[instance.method],
      'params': instance.params,
      'sender': instance.sender.toJson(),
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

const _$MethodEnumMap = {
  Method.create: 'create',
  Method.rerank: 'rerank',
  Method.faults: 'faults',
  Method.serialize: 'serialize',
  Method.analytics: 'analytics',
  Method.syncdataBytes: 'syncdataBytes',
  Method.synchronize: 'synchronize',
  Method.free: 'free',
};

CreateParams _$CreateParamsFromJson(Map json) => CreateParams(
      const Uint8ListConverter().fromJson(json['smbert_vocab'] as Uint8List),
      const Uint8ListConverter().fromJson(json['smbert_model'] as Uint8List),
      const Uint8ListConverter().fromJson(json['qambert_vocab'] as Uint8List),
      const Uint8ListConverter().fromJson(json['qambert_model'] as Uint8List),
      const Uint8ListConverter().fromJson(json['ltr_model'] as Uint8List),
      const Uint8ListConverter().fromJson(json['wasm_module'] as Uint8List),
      json['wasm_script'] as String,
      const Uint8ListMaybeNullConverter()
          .fromJson(json['serialized'] as Uint8List?),
    );

Map<String, dynamic> _$CreateParamsToJson(CreateParams instance) =>
    <String, dynamic>{
      'smbert_vocab': const Uint8ListConverter().toJson(instance.smbertVocab),
      'smbert_model': const Uint8ListConverter().toJson(instance.smbertModel),
      'qambert_vocab': const Uint8ListConverter().toJson(instance.qambertVocab),
      'qambert_model': const Uint8ListConverter().toJson(instance.qambertModel),
      'ltr_model': const Uint8ListConverter().toJson(instance.ltrModel),
      'wasm_module': const Uint8ListConverter().toJson(instance.wasmModule),
      'wasm_script': instance.wasmScript,
      'serialized':
          const Uint8ListMaybeNullConverter().toJson(instance.serialized),
    };

RerankParams _$RerankParamsFromJson(Map json) => RerankParams(
      _$enumDecode(_$RerankModeEnumMap, json['mode']),
      (json['histories'] as List<dynamic>)
          .map((e) => History.fromJson(Map<String, dynamic>.from(e as Map)))
          .toList(),
      (json['documents'] as List<dynamic>)
          .map((e) => Document.fromJson(Map<String, dynamic>.from(e as Map)))
          .toList(),
    );

Map<String, dynamic> _$RerankParamsToJson(RerankParams instance) =>
    <String, dynamic>{
      'mode': _$RerankModeEnumMap[instance.mode],
      'histories': instance.histories.map((e) => e.toJson()).toList(),
      'documents': instance.documents.map((e) => e.toJson()).toList(),
    };

const _$RerankModeEnumMap = {
  RerankMode.standardNews: 0,
  RerankMode.personalizedNews: 1,
  RerankMode.standardSearch: 2,
  RerankMode.personalizedSearch: 3,
};

SynchronizeParams _$SynchronizeParamsFromJson(Map json) => SynchronizeParams(
      const Uint8ListConverter().fromJson(json['serialized'] as Uint8List),
    );

Map<String, dynamic> _$SynchronizeParamsToJson(SynchronizeParams instance) =>
    <String, dynamic>{
      'serialized': const Uint8ListConverter().toJson(instance.serialized),
    };
