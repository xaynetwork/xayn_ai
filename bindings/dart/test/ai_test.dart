import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, isEmpty, test;

import 'package:xayn_ai_ffi_dart/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/error.dart' show XaynAiCode;
import 'utils.dart'
    show
        docsIds,
        docsRanks,
        docsSnippets,
        histFeedbacks,
        histIds,
        histRelevances,
        model,
        throwsXaynAiException,
        vocab;

void main() {
  group('XaynAi', () {
    test('new', () {
      final ai = XaynAi(vocab, model);
      ai.free();
    });

    test('rerank full', () {
      final ai = XaynAi(vocab, model);
      final reranked = ai.rerank(histIds, histRelevances, histFeedbacks,
          docsIds, docsSnippets, docsRanks);
      expect(reranked..sort(), equals(docsRanks));
      ai.free();
    });

    test('rerank empty', () {
      final ai = XaynAi(vocab, model);
      final reranked = ai.rerank([], [], [], [], [], []);
      expect(reranked, isEmpty);
      ai.free();
    });

    test('rerank empty hist', () {
      final ai = XaynAi(vocab, model);
      final reranked = ai.rerank([], [], [], docsIds, docsSnippets, docsRanks);
      expect(reranked..sort(), equals(docsRanks));
      ai.free();
    });

    test('rerank empty docs', () {
      final ai = XaynAi(vocab, model);
      final reranked =
          ai.rerank(histIds, histRelevances, histFeedbacks, [], [], []);
      expect(reranked, isEmpty);
      ai.free();
    });

    test('double free', () {
      final ai = XaynAi(vocab, model);
      ai.free();
      ai.free();
    });

    test('invalid paths', () {
      final code = XaynAiCode.readFile;
      final message =
          'Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)';

      expect(() => XaynAi('', model), throwsXaynAiException(code, message));
      expect(() => XaynAi(vocab, ''), throwsXaynAiException(code, message));
    });
  });
}
