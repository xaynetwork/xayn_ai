import 'package:flutter_test/flutter_test.dart'
    show Matcher, predicate, throwsA;

import 'package:xayn_ai_ffi_dart/src/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/data/history.dart'
    show Feedback, History, Relevance;
import 'package:xayn_ai_ffi_dart/src/result/error.dart'
    show Code, XaynAiException;

const vocab = '../../data/rubert_v0000/vocab.txt';
const model = '../../data/rubert_v0000/model.onnx';

final histories = [
  History('0', Relevance.low, Feedback.irrelevant),
  History('1', Relevance.high, Feedback.relevant),
];
final documents = [
  Document('0', 'abc', 0),
  Document('1', 'def', 1),
  Document('2', 'ghi', 2),
];

Matcher throwsXaynAiException(Code code, String message) =>
    throwsA(predicate((exception) =>
        exception is XaynAiException &&
        exception.code == code &&
        exception.toString() == message));
