import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show contains, equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart' show Code;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/data_provider.dart'
    show SetupData;
import '../utils.dart'
    show
        documents,
        histories,
        smbertModel,
        qambertModel,
        throwsXaynAiException,
        vocab;

void main() {
  group('XaynAi', () {
    test('rerank full', () {
      final ai = XaynAi(SetupData(vocab, smbertModel, qambertModel));
      final outcome = ai.rerank(histories, documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty', () {
      final ai = XaynAi(SetupData(vocab, smbertModel, qambertModel));
      final outcome = ai.rerank([], []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty hists', () {
      final ai = XaynAi(SetupData(vocab, smbertModel, qambertModel));
      final outcome = ai.rerank([], documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty docs', () {
      final ai = XaynAi(SetupData(vocab, smbertModel, qambertModel));
      final outcome = ai.rerank(histories, []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('invalid paths', () {
      expect(
        () => XaynAi(SetupData('', smbertModel, qambertModel)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () => XaynAi(SetupData(vocab, '', qambertModel)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () => XaynAi(SetupData(vocab, smbertModel, '')),
        throwsXaynAiException(Code.readFile),
      );
    });

    test('empty serialized', () {
      final serialized = Uint8List(0);
      final ai =
          XaynAi(SetupData(vocab, smbertModel, qambertModel), serialized);
      ai.free();
    });

    test('invalid serialized', () {
      expect(
        () => XaynAi(SetupData(vocab, smbertModel, qambertModel),
            Uint8List.fromList([255])),
        throwsXaynAiException(Code.rerankerDeserialization),
      );
    });
  });
}
