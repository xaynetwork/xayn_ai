import 'package:flutter_test/flutter_test.dart'
    show Matcher, predicate, throwsA;

import 'package:xayn_ai_ffi_dart/src/doc/document.dart'
    show Document, Feedback, History, Relevance;
import 'package:xayn_ai_ffi_dart/src/error.dart'
    show XaynAiCode, XaynAiException;

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

Matcher throwsXaynAiException(XaynAiCode code, String message) =>
    throwsA(predicate((exception) =>
        exception is XaynAiException &&
        exception.code == code &&
        exception.toString() == message));
