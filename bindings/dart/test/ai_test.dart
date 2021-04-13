import 'package:flutter_test/flutter_test.dart'
    show contains, equals, expect, group, isEmpty, test;

import 'package:xayn_ai_ffi_dart/src/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/src/error.dart' show XaynAiCode;
import 'utils.dart'
    show documents, histories, model, throwsXaynAiException, vocab;

void main() {
  group('XaynAi', () {
    test('rerank full', () {
      final ai = XaynAi(vocab, model);
      final ranks = ai.rerank(histories, documents);
      expect(ranks.length, equals(documents.length));
      documents.forEach((document) => expect(ranks, contains(document.rank)));
      ai.free();
    });

    test('rerank empty', () {
      final ai = XaynAi(vocab, model);
      final ranks = ai.rerank([], []);
      expect(ranks, isEmpty);
      ai.free();
    });

    test('rerank empty hists', () {
      final ai = XaynAi(vocab, model);
      final ranks = ai.rerank([], documents);
      expect(ranks.length, equals(documents.length));
      documents.forEach((document) => expect(ranks, contains(document.rank)));
      ai.free();
    });

    test('rerank empty docs', () {
      final ai = XaynAi(vocab, model);
      final ranks = ai.rerank(histories, []);
      expect(ranks, isEmpty);
      ai.free();
    });

    test('invalid paths', () {
      final code = XaynAiCode.readFile;
      final message =
          'Failed to initialize the ai: Failed to load a data file: No such file or directory (os error 2)';
      expect(() => XaynAi('', model), throwsXaynAiException(code, message));
      expect(() => XaynAi(vocab, ''), throwsXaynAiException(code, message));
    });
  });
}
