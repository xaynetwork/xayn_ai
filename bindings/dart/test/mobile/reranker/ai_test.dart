import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show contains, equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart' show Code;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/ai.dart' show createXaynAi;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/data_provider.dart'
    show SetupData;
import '../utils.dart'
    show documents, histories, model, throwsXaynAiException, vocab;

void main() {
  group('XaynAi', () {
    test('rerank full', () async {
      final ai = await createXaynAi(SetupData(vocab, model));
      final outcome = ai.rerank(histories, documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty', () async {
      final ai = await createXaynAi(SetupData(vocab, model));
      final outcome = ai.rerank([], []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty hists', () async {
      final ai = await createXaynAi(SetupData(vocab, model));
      final outcome = ai.rerank([], documents);
      final faults = ai.faults();

      expect(outcome.finalRanks.length, equals(documents.length));
      documents.forEach(
          (document) => expect(outcome.finalRanks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty docs', () async {
      final ai = await createXaynAi(SetupData(vocab, model));
      final outcome = ai.rerank(histories, []);
      final faults = ai.faults();

      expect(outcome.finalRanks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('invalid paths', () {
      expect(
        () async => await createXaynAi(SetupData('', model)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () async => await createXaynAi(SetupData(vocab, '')),
        throwsXaynAiException(Code.readFile),
      );
    });

    test('empty serialized', () async {
      final serialized = Uint8List(0);
      final ai = await createXaynAi(SetupData(vocab, model), serialized);
      ai.free();
    });

    test('invalid serialized', () {
      expect(
        () async => await createXaynAi(
            SetupData(vocab, model), Uint8List.fromList([255])),
        throwsXaynAiException(Code.rerankerDeserialization),
      );
    });
  });
}
