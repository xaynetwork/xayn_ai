import 'package:flutter_test/flutter_test.dart'
    show Matcher, predicate, throwsA;

import 'package:xayn_ai_ffi_dart/error.dart' show XaynAiCode, XaynAiException;

const vocab = '../../data/rubert_v0000/vocab.txt';
const model = '../../data/rubert_v0000/model.onnx';

Matcher throwsXaynAiException(XaynAiCode code, String message) =>
    throwsA(predicate((exc) =>
        exc is XaynAiException &&
        exc.toString() == '${code.toString().split('.').last}: $message'));
