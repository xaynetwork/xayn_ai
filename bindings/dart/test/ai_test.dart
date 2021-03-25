import 'package:flutter_test/flutter_test.dart'
    show equals, expect, group, Matcher, predicate, test, throwsA;

import 'package:xayn_ai_ffi_dart/ai.dart' show XaynAi;
import 'package:xayn_ai_ffi_dart/error.dart';

void main() {
  const vocab = '../../data/rubert_v0000/vocab.txt';
  const model = '../../data/rubert_v0000/model.onnx';

  Matcher throwsXaynAiException(String message) => throwsA(
      predicate((exc) => exc is XaynAiException && exc.toString() == message));

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
      final message =
          'ReadFile: Failed to build the bert model: Failed to load a data file: No such file or directory (os error 2)';
      expect(() => XaynAi('', model), throwsXaynAiException(message));
      expect(() => XaynAi(vocab, ''), throwsXaynAiException(message));
    });
  });
}
