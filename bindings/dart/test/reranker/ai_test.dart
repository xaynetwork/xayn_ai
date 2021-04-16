import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show contains, equals, expect, group, isEmpty, isNot, test;

import 'package:xayn_ai_ffi_dart/src/reranker/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/src/result/error.dart' show Code;
import '../utils.dart'
    show documents, histories, model, throwsXaynAiException, vocab;

void main() {
  group('XaynAi', () {
    test('rerank full', () {
      final ai = XaynAi(Uint8List(0), vocab, model);
      final ranks = ai.rerank(histories, documents);
      final warnings = ai.warnings();

      expect(ranks.length, equals(documents.length));
      documents.forEach((document) => expect(ranks, contains(document.rank)));
      expect(warnings, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty', () {
      final ai = XaynAi(Uint8List(0), vocab, model);
      final ranks = ai.rerank([], []);
      final warnings = ai.warnings();

      expect(ranks, isEmpty);
      expect(warnings, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty hists', () {
      final ai = XaynAi(Uint8List(0), vocab, model);
      final ranks = ai.rerank([], documents);
      final warnings = ai.warnings();

      expect(ranks.length, equals(documents.length));
      documents.forEach((document) => expect(ranks, contains(document.rank)));
      expect(warnings, isNot(isEmpty));

      ai.free();
    });

    test('rerank empty docs', () {
      final ai = XaynAi(Uint8List(0), vocab, model);
      final ranks = ai.rerank(histories, []);
      final warnings = ai.warnings();

      expect(ranks, isEmpty);
      expect(warnings, isNot(isEmpty));

      ai.free();
    });

    test('invalid paths', () {
      final code = Code.readFile;
      final message =
          'Failed to initialize the ai: Failed to load a data file: No such file or directory (os error 2)';
      expect(() => XaynAi(Uint8List(0), '', model),
          throwsXaynAiException(code, message));
      expect(() => XaynAi(Uint8List(0), vocab, ''),
          throwsXaynAiException(code, message));
    });
  });
}
