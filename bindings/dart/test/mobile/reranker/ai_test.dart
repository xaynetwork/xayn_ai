import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show contains, equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart' show Code;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/data_provider.dart'
    show SetupData;
import '../utils.dart'
    show documents, histories, model, throwsXaynAiException, vocab;

void main() {
  group('XaynAi', () {
    test('rerank full', () {
      final ai = XaynAi(SetupData(vocab, model));
      final ranks = ai.rerank(histories, documents);
      final faults = ai.faults();

      expect(ranks.length, equals(documents.length));
      documents.forEach((document) => expect(ranks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty', () {
      final ai = XaynAi(SetupData(vocab, model));
      final ranks = ai.rerank([], []);
      final faults = ai.faults();

      expect(ranks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty hists', () {
      final ai = XaynAi(SetupData(vocab, model));
      final ranks = ai.rerank([], documents);
      final faults = ai.faults();

      expect(ranks.length, equals(documents.length));
      documents.forEach((document) => expect(ranks, contains(document.rank)));
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty docs', () {
      final ai = XaynAi(SetupData(vocab, model));
      final ranks = ai.rerank(histories, []);
      final faults = ai.faults();

      expect(ranks, isEmpty);
      expect(faults, isNot(isEmpty));

      ai.free();
    });

    test('invalid paths', () {
      expect(
        () => XaynAi(SetupData('', model)),
        throwsXaynAiException(Code.readFile),
      );
      expect(
        () => XaynAi(SetupData(vocab, '')),
        throwsXaynAiException(Code.readFile),
      );
    });

    test('empty serialized', () {
      final serialized = Uint8List(0);
      final ai = XaynAi(SetupData(vocab, model), serialized);
      ai.free();
    });

    test('invalid serialized', () {
      expect(
        () => XaynAi(SetupData(vocab, model), Uint8List.fromList([255])),
        throwsXaynAiException(Code.rerankerDeserialization),
      );
    });
  });
}