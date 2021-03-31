import 'package:flutter_test/flutter_test.dart'
    show Matcher, predicate, throwsA;

import 'package:xayn_ai_ffi_dart/document.dart' show Feedback, Relevance;
import 'package:xayn_ai_ffi_dart/error.dart' show XaynAiCode, XaynAiException;

const vocab = '../../data/rubert_v0000/vocab.txt';
const model = '../../data/rubert_v0000/model.onnx';

const histIds = ['0', '1'];
const histRelevances = [Relevance.low, Relevance.high];
const histFeedbacks = [Feedback.irrelevant, Feedback.relevant];

const docsIds = ['0', '1', '2'];
const docsSnippets = ['abc', 'def', 'ghi'];
const docsRanks = [0, 1, 2];

Matcher throwsXaynAiException(XaynAiCode code, String message) =>
    throwsA(predicate((exc) =>
        exc is XaynAiException &&
        exc.toString() == '${code.toString().split('.').last}: $message'));
