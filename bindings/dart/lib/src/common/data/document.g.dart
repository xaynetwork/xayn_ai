// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'document.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Document _$DocumentFromJson(Map json) => Document(
      id: json['id'] as String,
      title: json['title'] as String,
      snippet: json['snippet'] as String,
      rank: json['rank'] as int,
      session: json['session'] as String,
      queryCount: json['query_count'] as int,
      queryId: json['query_id'] as String,
      queryWords: json['query_words'] as String,
      url: json['url'] as String,
      domain: json['domain'] as String,
    );

Map<String, dynamic> _$DocumentToJson(Document instance) => <String, dynamic>{
      'id': instance.id,
      'title': instance.title,
      'snippet': instance.snippet,
      'rank': instance.rank,
      'session': instance.session,
      'query_count': instance.queryCount,
      'query_id': instance.queryId,
      'query_words': instance.queryWords,
      'url': instance.url,
      'domain': instance.domain,
    };
