import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show contains, equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart' show Code;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import '../utils.dart'
    show
        documents,
        histories,
        mkSetupData,
        smbertModel,
        smbertVocab,
        qambertModel,
        qambertVocab,
        ltrModel,
        throwsXaynAiException;

void main() {
  group('XaynAi', () {
    test('rerank full', () async {
      final ai = await XaynAi.create(mkSetupData(
          smbertVocab, smbertModel, qambertVocab, qambertModel, ltrModel));
      final outcome = ai.rerank(RerankMode.search, histories, documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty', () async {
      final ai = await XaynAi.create(mkSetupData(
          smbertVocab, smbertModel, qambertVocab, qambertModel, ltrModel));
      final outcome = ai.rerank(RerankMode.search, [], []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty hists', () async {
      final ai = await XaynAi.create(mkSetupData(
          smbertVocab, smbertModel, qambertVocab, qambertModel, ltrModel));
      final outcome = ai.rerank(RerankMode.search, [], documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty docs', () async {
      final ai = await XaynAi.create(mkSetupData(
          smbertVocab, smbertModel, qambertVocab, qambertModel, ltrModel));
      final outcome = ai.rerank(RerankMode.search, histories, []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('invalid paths', () {
      expect(
        () async => await XaynAi.create(
            mkSetupData('', smbertModel, qambertVocab, qambertModel, ltrModel)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async => await XaynAi.create(
            mkSetupData(smbertVocab, '', qambertVocab, qambertModel, ltrModel)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async => await XaynAi.create(
            mkSetupData(smbertVocab, smbertModel, '', qambertModel, ltrModel)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async => await XaynAi.create(
            mkSetupData(smbertVocab, smbertModel, qambertVocab, '', ltrModel)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async => await XaynAi.create(mkSetupData(
            smbertVocab, smbertModel, qambertVocab, qambertModel, '')),
        throwsXaynAiException(Code.readFile),
      );
    });

    test('empty serialized', () async {
      final serialized = Uint8List(0);
      final ai = await XaynAi.create(
          mkSetupData(
              smbertVocab, smbertModel, qambertVocab, qambertModel, ltrModel),
          serialized);
      ai.free();
    });

    test('invalid serialized', () {
      expect(
        () async => await XaynAi.create(
            mkSetupData(
                smbertVocab, smbertModel, qambertVocab, qambertModel, ltrModel),
            Uint8List.fromList([255])),
        throwsXaynAiException(Code.rerankerDeserialization),
      );
    });
  });
}
