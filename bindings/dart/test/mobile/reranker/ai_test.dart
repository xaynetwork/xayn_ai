import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show contains, equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart' show Code;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/ai.dart' show createXaynAi;
import '../utils.dart'
    show
        documents,
        histories,
        mkSetupData,
        smbertModel,
        smbertVocab,
        qambertModel,
        qambertVocab,
        throwsXaynAiException;

void main() {
  group('XaynAi', () {
    test('rerank full', () async {
      final ai = await createXaynAi(
          mkSetupData(smbertVocab, smbertModel, qambertVocab, qambertModel));
      final outcome = ai.rerank(histories, documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty', () async {
      final ai = await createXaynAi(
          mkSetupData(smbertVocab, smbertModel, qambertVocab, qambertModel));
      final outcome = ai.rerank([], []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty hists', () async {
      final ai = await createXaynAi(
          mkSetupData(smbertVocab, smbertModel, qambertVocab, qambertModel));
      final outcome = ai.rerank([], documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty docs', () async {
      final ai = await createXaynAi(
          mkSetupData(smbertVocab, smbertModel, qambertVocab, qambertModel));
      final outcome = ai.rerank(histories, []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('invalid paths', () {
      expect(
        () async => await createXaynAi(
            mkSetupData('', smbertModel, qambertVocab, qambertModel)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async => await createXaynAi(
            mkSetupData(smbertVocab, '', qambertVocab, qambertModel)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async => await createXaynAi(
            mkSetupData(smbertVocab, smbertModel, '', qambertModel)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async => await createXaynAi(
            mkSetupData(smbertVocab, smbertModel, qambertVocab, '')),
        throwsXaynAiException(Code.readFile),
      );
    });

    test('empty serialized', () async {
      final serialized = Uint8List(0);
      final ai = await createXaynAi(
          mkSetupData(smbertVocab, smbertModel, qambertVocab, qambertModel),
          serialized);
      ai.free();
    });

    test('invalid serialized', () {
      expect(
        () async => await createXaynAi(
            mkSetupData(smbertVocab, smbertModel, qambertVocab, qambertModel),
            Uint8List.fromList([255])),
        throwsXaynAiException(Code.rerankerDeserialization),
      );
    });
  });
}
