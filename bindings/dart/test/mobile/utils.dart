import 'package:flutter_test/flutter_test.dart'
    show Matcher, predicate, throwsA;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show UserFeedback, History, Relevance, DayOfWeek, UserAction;
import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show Code, XaynAiException;
import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    show AssetType;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/data_provider.dart'
    show SetupData;

const smbertVocab = '../../data/smbert_v0000/vocab.txt';
const smbertModel = '../../data/smbert_v0000/smbert.onnx';
const qambertVocab = '../../data/qambert_v0001/vocab.txt';
const qambertModel = '../../data/qambert_v0001/qambert.onnx';

SetupData mkSetupData(String smbertVocab, String smbertModel,
    String qambertVocab, String qambertModel) {
  return SetupData(<AssetType, String>{
    AssetType.smbertVocab: smbertVocab,
    AssetType.smbertModel: smbertModel,
    AssetType.qambertVocab: qambertVocab,
    AssetType.qambertModel: qambertModel,
  });
}

Document mkTestDoc(String id, String snippet, int rank) => Document(
      id: id,
      snippet: snippet,
      rank: rank,
      session: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryCount: 1,
      queryId: 'fcb6a685-eb92-4d36-8686-000000000000',
      queryWords: 'query words',
      url: 'url',
      domain: 'domain',
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
