import 'dart:typed_data' show Uint8List;

import 'package:flutter_test/flutter_test.dart'
    show contains, equals, expect, group, isEmpty, test;

import 'package:xayn_ai_ffi_dart/src/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/src/error.dart' show Code;
import 'utils.dart'
    show documents, histories, model, throwsXaynAiException, vocab;

void main() {
  group('XaynAi', () {
    test('rerank full', () {
      final ai = XaynAi(Uint8List(0), vocab, model);
      final ranks = ai.rerank(histories, documents);
      expect(ranks.length, equals(documents.length));
      documents.forEach((document) => expect(ranks, contains(document.rank)));
      ai.free();
    });

    test('rerank empty', () {
      final ai = XaynAi(Uint8List(0), vocab, model);
      final ranks = ai.rerank([], []);
      expect(ranks, isEmpty);
      ai.free();
    });

    test('rerank empty hists', () {
      final ai = XaynAi(Uint8List(0), vocab, model);
      final ranks = ai.rerank([], documents);
      expect(ranks.length, equals(documents.length));
      documents.forEach((document) => expect(ranks, contains(document.rank)));
      ai.free();
    });

    test('rerank empty docs', () {
      final ai = XaynAi(Uint8List(0), vocab, model);
      final ranks = ai.rerank(histories, []);
      expect(ranks, isEmpty);
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
