import 'package:flutter_test/flutter_test.dart'
    show Matcher, predicate, throwsA;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show UserFeedback, History, Relevance, DayOfWeek, UserAction;
import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show Code, XaynAiException;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/data_provider.dart'
    show getAssets, SetupData;

SetupData mkSetupData() {
  return SetupData({
    for (final asset in getAssets().entries)
      asset.key: '../../data/' + asset.value.urlSuffix
  });
}

Document mkTestDoc(String id, String title, int rank) => Document(
      id: id,
      title: title,
      snippet: 'snippet of $title',
      rank: rank,
      session: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryCount: 1,
      queryId: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryWords: 'query words',
      url: 'url',
      domain: 'domain',
      viewed: 0,
    );

History mkTestHist(String id, Relevance relevance, UserFeedback feedback) =>
    History(
      id: id,
      relevance: relevance,
      userFeedback: feedback,
      session: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryCount: 1,
      queryId: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryWords: 'query words',
      day: DayOfWeek.mon,
      url: 'url',
      domain: 'domain',
      rank: 0,
      userAction: UserAction.miss,
    );

final histories = [
  mkTestHist('fcb6a685-eb92-4d36-8686-000000000000', Relevance.low,
      UserFeedback.irrelevant),
  mkTestHist('fcb6a685-eb92-4d36-8686-000000000001', Relevance.high,
      UserFeedback.relevant),
];

final documents = [
  mkTestDoc('fcb6a685-eb92-4d36-8686-000000000000', 'abc', 0),
  mkTestDoc('fcb6a685-eb92-4d36-8686-000000000001', 'def', 1),
  mkTestDoc('fcb6a685-eb92-4d36-8686-000000000002', 'ghi', 2),
];

Matcher throwsXaynAiException(Code code) => throwsA(
      predicate(
        (exception) =>
            exception is XaynAiException &&
            exception.code == code &&
            exception.toString().isNotEmpty,
      ),
    );
