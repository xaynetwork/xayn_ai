// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'error.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

XaynAiException _$XaynAiExceptionFromJson(Map json) => XaynAiException(
      _$enumDecode(_$CodeEnumMap, json['code']),
      json['message'] as String,
    );

Map<String, dynamic> _$XaynAiExceptionToJson(XaynAiException instance) =>
    <String, dynamic>{
      'code': _$CodeEnumMap[instance.code],
      'message': instance.message,
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

const _$CodeEnumMap = {
  Code.fault: 'fault',
  Code.panic: 'panic',
  Code.none: 'none',
  Code.smbertVocabPointer: 'smbertVocabPointer',
  Code.smbertModelPointer: 'smbertModelPointer',
  Code.qambertVocabPointer: 'qambertVocabPointer',
  Code.qambertModelPointer: 'qambertModelPointer',
  Code.readFile: 'readFile',
  Code.initAi: 'initAi',
  Code.aiPointer: 'aiPointer',
  Code.historiesPointer: 'historiesPointer',
  Code.historyIdPointer: 'historyIdPointer',
  Code.historySessionPointer: 'historySessionPointer',
  Code.historyQueryIdPointer: 'historyQueryIdPointer',
  Code.historyQueryWordsPointer: 'historyQueryWordsPointer',
  Code.historyUrlPointer: 'historyUrlPointer',
  Code.historyDomainPointer: 'historyDomainPointer',
  Code.documentsPointer: 'documentsPointer',
  Code.documentIdPointer: 'documentIdPointer',
  Code.documentTitlePointer: 'documentTitlePointer',
  Code.documentSnippetPointer: 'documentSnippetPointer',
  Code.documentSessionPointer: 'documentSessionPointer',
  Code.documentQueryIdPointer: 'documentQueryIdPointer',
  Code.documentQueryWordsPointer: 'documentQueryWordsPointer',
  Code.documentUrlPointer: 'documentUrlPointer',
  Code.documentDomainPointer: 'documentDomainPointer',
  Code.rerankerDeserialization: 'rerankerDeserialization',
  Code.rerankerSerialization: 'rerankerSerialization',
  Code.historiesDeserialization: 'historiesDeserialization',
  Code.documentsDeserialization: 'documentsDeserialization',
  Code.rerankModeDeserialization: 'rerankModeDeserialization',
  Code.syncDataSerialization: 'syncDataSerialization',
  Code.synchronization: 'synchronization',
  Code.syncDataBytesPointer: 'syncDataBytesPointer',
  Code.initGlobalThreadPool: 'initGlobalThreadPool',
};
