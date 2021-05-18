import 'package:flutter_test/flutter_test.dart'
    show Matcher, predicate, throwsA;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show Feedback, History, Relevance;
import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show Code, XaynAiException;

const vocab = '../../data/rubert_v0001/vocab.txt';
const model = '../../data/rubert_v0001/smbert.onnx';

final histories = [
  History('00000000-0000-0000-0000-000000000000', Relevance.low,
      Feedback.irrelevant),
  History('00000000-0000-0000-0000-000000000001', Relevance.high,
      Feedback.relevant),
];
final documents = [
  Document('00000000-0000-0000-0000-000000000000', 'abc', 0),
  Document('00000000-0000-0000-0000-000000000001', 'def', 1),
  Document('00000000-0000-0000-0000-000000000002', 'ghi', 2),
];

Matcher throwsXaynAiException(Code code) => throwsA(
      predicate(
        (exception) =>
            exception is XaynAiException &&
            exception.code == code &&
            exception.toString().isNotEmpty,
      ),
    );
