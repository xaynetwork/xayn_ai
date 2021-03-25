import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, test;

import 'package:xayn_ai_ffi_dart/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/error.dart' show XaynAiCode;
import 'utils.dart' show model, throwsXaynAiException, vocab;

void main() {
  group('XaynAi', () {
    test('new', () {
      final ai = XaynAi(vocab, model);
      ai.free();
    });

    test('rerank', () {
      final ai = XaynAi(vocab, model);
      final ranks = [0, 1, 2];
      final reranks = List.from(ranks.reversed, growable: false);

      final reranked = ai.rerank(['0', '1', '2'], ['abc', 'def', 'ghi'], ranks);
      expect(reranked, equals(reranks));

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
