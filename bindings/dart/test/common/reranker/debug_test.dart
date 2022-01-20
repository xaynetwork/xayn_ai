import 'dart:convert' show JsonEncoder, JsonDecoder;
import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isNull, test;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show History, Relevance, UserFeedback, UserAction, DayOfWeek;
import 'package:xayn_ai_ffi_dart/src/common/reranker/debug.dart'
    show RerankDebugCallData;
import 'package:xayn_ai_ffi_dart/src/common/reranker/mode.dart'
    show RerankMode, RerankModeToInt;

String encodeJson(Map<String, dynamic> object) => JsonEncoder().convert(object);

Map<String, dynamic> decodeJson(String json) =>
    JsonDecoder().convert(json) as Map<String, dynamic>;

void main() {
  group('RerankerDebugCallData', () {
    test('json_serialization no data', () {
      final histories = List<History>.empty();
      final documents = List<Document>.empty();
      final Uint8List? serializedState = null;
      final jsonMap = RerankDebugCallData(
        rerankMode: RerankMode.personalizedSearch,
        histories: histories,
        documents: documents,
        serializedState: serializedState,
      ).toJson();

      expect(jsonMap['serialized_state'], isNull);

      final deCallData = RerankDebugCallData.fromJson(jsonMap);

      expect(deCallData.rerankMode, equals(RerankMode.personalizedSearch));
      expect(deCallData.documents.length, equals(0));
      expect(deCallData.histories.length, equals(0));
      expect(deCallData.serializedState, isNull);
    });

    test('json_serialization', () {
      final histories = [
        History(
          id: 'fcb6a685-eb92-4d36-8686-8a70a3a33000',
          relevance: Relevance.low,
          userFeedback: UserFeedback.notGiven,
          session: 'fcb6a685-eb92-4d36-8686-000000000000',
          queryCount: 12,
          queryId: 'fcb6a685-eb92-4d36-8686-A000A00A000A',
          queryWords: 'is the dodo alive',
          day: DayOfWeek.sun,
          url: 'dodo lives:or not',
          domain: 'no domain',
          rank: 8,
          userAction: UserAction.click,
        ),
        History(
          id: 'fcb6a685-eb92-4d36-8686-8a70a3a33001',
          relevance: Relevance.high,
          userFeedback: UserFeedback.irrelevant,
          session: 'fcb6a685-eb92-4d36-8686-000000000000',
          queryCount: 12,
          queryId: 'fcb6a685-eb92-4d36-8686-A000A00A000A',
          queryWords: 'is the dodo alive',
          day: DayOfWeek.tue,
          url: 'dodo lives:or not',
          domain: 'no domain',
          rank: 8,
          userAction: UserAction.skip,
        ),
        History(
          id: 'fcb6a685-eb92-4d36-8686-8a70a3a33002',
          relevance: Relevance.medium,
          userFeedback: UserFeedback.relevant,
          session: 'fcb6a685-eb92-4d36-8686-000000000000',
          queryCount: 12,
          queryId: 'fcb6a685-eb92-4d36-8686-A000A00A000A',
          queryWords: 'is the dodo alive',
          day: DayOfWeek.tue,
          url: 'dodo lives:or not',
          domain: 'no domain',
          rank: 8,
          userAction: UserAction.miss,
        ),
      ];
      final documents = [
        Document(
          id: 'fcb6a685-eb92-4d36-8686-8a70a3a33003',
          title: 'a b c',
          snippet: 'snippet of a b c',
          rank: 1,
          session: 'fcb6a685-eb92-4d36-8686-000000000000',
          queryCount: 21,
          queryId: 'fcb6a685-eb92-4d36-8686-A000A00B000B',
          queryWords: 'abc',
          url: 'url',
          domain: 'dom',
          viewed: 0,
        ),
        Document(
          id: 'fcb6a685-eb92-4d36-8686-8a70a3a33004',
          title: 'ab de',
          snippet: 'snippet of ab de',
          rank: 2,
          session: 'fcb6a685-eb92-4d36-8686-000000000100',
          queryCount: 21,
          queryId: 'fcb6a685-eb92-4d36-8686-A000A00B000B',
          queryWords: 'abc',
          url: 'url2',
          domain: 'dom2',
          viewed: 0,
        ),
      ];

      final rerankMode = RerankMode.personalizedSearch;
      final serializedState = Uint8List.fromList([1, 2, 3, 4, 5, 6, 10, 20, 7]);

      final jsonMap = RerankDebugCallData(
        rerankMode: rerankMode,
        histories: histories,
        documents: documents,
        serializedState: serializedState,
      ).toJson();

      expect(jsonMap['rerank_mode'], equals(rerankMode.toInt()));

      expect(jsonMap['histories'][1]['id'],
          equals('fcb6a685-eb92-4d36-8686-8a70a3a33001'));
      expect(jsonMap['histories'][1]['relevance'], equals(2));
      expect(jsonMap['histories'][1]['user_feedback'], equals(1));
      expect(jsonMap['histories'][1]['session'],
          equals('fcb6a685-eb92-4d36-8686-000000000000'));
      expect(jsonMap['histories'][1]['query_count'], equals(12));
      expect(jsonMap['histories'][1]['query_id'],
          equals('fcb6a685-eb92-4d36-8686-A000A00A000A'));
      expect(
          jsonMap['histories'][1]['query_words'], equals('is the dodo alive'));
      expect(jsonMap['histories'][1]['day'], equals(1));
      expect(jsonMap['histories'][1]['url'], equals('dodo lives:or not'));
      expect(jsonMap['histories'][1]['domain'], equals('no domain'));
      expect(jsonMap['histories'][1]['rank'], equals(8));
      expect(jsonMap['histories'][1]['user_action'], equals(1));

      expect(jsonMap['documents'][1]['id'],
          equals('fcb6a685-eb92-4d36-8686-8a70a3a33004'));
      expect(jsonMap['documents'][1]['title'], equals('ab de'));
      expect(jsonMap['documents'][1]['snippet'], equals('snippet of ab de'));
      expect(jsonMap['documents'][1]['rank'], equals(2));
      expect(jsonMap['documents'][1]['session'],
          equals('fcb6a685-eb92-4d36-8686-000000000100'));
      expect(jsonMap['documents'][1]['query_count'], equals(21));
      expect(jsonMap['documents'][1]['query_id'],
          equals('fcb6a685-eb92-4d36-8686-A000A00B000B'));
      expect(jsonMap['documents'][1]['query_words'], equals('abc'));
      expect(jsonMap['documents'][1]['url'], equals('url2'));
      expect(jsonMap['documents'][1]['domain'], equals('dom2'));
      expect(jsonMap['documents'][1]['viewed'], equals(0));

      expect(jsonMap['serialized_state'], equals('AQIDBAUGChQH'));

      final callData = RerankDebugCallData.fromJson(jsonMap);

      expect(callData.rerankMode, equals(RerankMode.personalizedSearch));

      expect(callData.histories[0].id,
          equals('fcb6a685-eb92-4d36-8686-8a70a3a33000'));
      expect(callData.histories[0].relevance, equals(Relevance.low));
      expect(callData.histories[0].userFeedback, equals(UserFeedback.notGiven));
      expect(callData.histories[0].session,
          equals('fcb6a685-eb92-4d36-8686-000000000000'));
      expect(callData.histories[0].queryCount, equals(12));
      expect(callData.histories[0].queryId,
          equals('fcb6a685-eb92-4d36-8686-A000A00A000A'));
      expect(callData.histories[0].queryWords, equals('is the dodo alive'));
      expect(callData.histories[0].day, equals(DayOfWeek.sun));
      expect(callData.histories[0].url, equals('dodo lives:or not'));
      expect(callData.histories[0].domain, equals('no domain'));
      expect(callData.histories[0].rank, equals(8));
      expect(callData.histories[0].userAction, equals(UserAction.click));

      expect(callData.histories[1].id,
          equals('fcb6a685-eb92-4d36-8686-8a70a3a33001'));
      expect(callData.histories[1].relevance, equals(Relevance.high));
      expect(
          callData.histories[1].userFeedback, equals(UserFeedback.irrelevant));
      expect(callData.histories[1].session,
          equals('fcb6a685-eb92-4d36-8686-000000000000'));
      expect(callData.histories[1].queryCount, equals(12));
      expect(callData.histories[1].queryId,
          equals('fcb6a685-eb92-4d36-8686-A000A00A000A'));
      expect(callData.histories[1].queryWords, equals('is the dodo alive'));
      expect(callData.histories[1].day, equals(DayOfWeek.tue));
      expect(callData.histories[1].url, equals('dodo lives:or not'));
      expect(callData.histories[1].domain, equals('no domain'));
      expect(callData.histories[1].rank, equals(8));
      expect(callData.histories[1].userAction, equals(UserAction.skip));

      expect(callData.histories[2].id,
          equals('fcb6a685-eb92-4d36-8686-8a70a3a33002'));
      expect(callData.histories[2].relevance, equals(Relevance.medium));
      expect(callData.histories[2].userFeedback, equals(UserFeedback.relevant));
      expect(callData.histories[2].session,
          equals('fcb6a685-eb92-4d36-8686-000000000000'));
      expect(callData.histories[2].queryCount, equals(12));
      expect(callData.histories[2].queryId,
          equals('fcb6a685-eb92-4d36-8686-A000A00A000A'));
      expect(callData.histories[2].queryWords, equals('is the dodo alive'));
      expect(callData.histories[2].day, equals(DayOfWeek.tue));
      expect(callData.histories[2].url, equals('dodo lives:or not'));
      expect(callData.histories[2].domain, equals('no domain'));
      expect(callData.histories[2].rank, equals(8));
      expect(callData.histories[2].userAction, equals(UserAction.miss));

      expect(callData.histories.length, equals(3));

      expect(callData.documents[0].id,
          equals('fcb6a685-eb92-4d36-8686-8a70a3a33003'));
      expect(callData.documents[0].title, equals('a b c'));
      expect(callData.documents[0].snippet, equals('snippet of a b c'));
      expect(callData.documents[0].rank, equals(1));
      expect(callData.documents[0].session,
          equals('fcb6a685-eb92-4d36-8686-000000000000'));
      expect(callData.documents[0].queryCount, equals(21));
      expect(callData.documents[0].queryId,
          equals('fcb6a685-eb92-4d36-8686-A000A00B000B'));
      expect(callData.documents[0].queryWords, equals('abc'));
      expect(callData.documents[0].url, equals('url'));
      expect(callData.documents[0].domain, equals('dom'));
      expect(callData.documents[0].viewed, equals(0));

      expect(callData.documents[1].id,
          equals('fcb6a685-eb92-4d36-8686-8a70a3a33004'));
      expect(callData.documents[1].title, equals('ab de'));
      expect(callData.documents[1].snippet, equals('snippet of ab de'));
      expect(callData.documents[1].rank, equals(2));
      expect(callData.documents[1].session,
          equals('fcb6a685-eb92-4d36-8686-000000000100'));
      expect(callData.documents[1].queryCount, equals(21));
      expect(callData.documents[1].queryId,
          equals('fcb6a685-eb92-4d36-8686-A000A00B000B'));
      expect(callData.documents[1].queryWords, equals('abc'));
      expect(callData.documents[1].url, equals('url2'));
      expect(callData.documents[1].domain, equals('dom2'));
      expect(callData.documents[1].viewed, equals(0));

      expect(callData.documents.length, equals(2));

      expect(callData.serializedState, equals([1, 2, 3, 4, 5, 6, 10, 20, 7]));
    });

    test('serialized_state defaults to null', () {
      final callData = RerankDebugCallData(
        rerankMode: RerankMode.personalizedSearch,
        histories: [],
        documents: [],
      );
      expect(callData.rerankMode, equals(RerankMode.personalizedSearch));
      expect(callData.serializedState, isNull);

      final jsonMap = RerankDebugCallData(
        rerankMode: RerankMode.personalizedNews,
        histories: [],
        documents: [],
        serializedState: Uint8List.fromList([1, 2, 3]),
      ).toJson();

      jsonMap.remove('serialized_state');

      final newCallData = RerankDebugCallData.fromJson(jsonMap);
      expect(newCallData.serializedState, isNull);
    });
  });
}
