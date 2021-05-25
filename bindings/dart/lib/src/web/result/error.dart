@JS()
library error;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show Code, XaynAiException;

extension CodeToInt on Code {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Code.fault:
        return -2;
      case Code.panic:
        return -1;
      case Code.none:
        return 0;
      case Code.vocabPointer:
        return 1;
      case Code.modelPointer:
        return 2;
      case Code.readFile:
        return 3;
      case Code.initAi:
        return 4;
      case Code.aiPointer:
        return 5;
      case Code.historiesPointer:
        return 6;
      case Code.historyIdPointer:
        return 7;
      case Code.documentsPointer:
        return 8;
      case Code.documentIdPointer:
        return 9;
      case Code.documentSnippetPointer:
        return 10;
      case Code.rerankerDeserialization:
        return 11;
      case Code.rerankerSerialization:
        return 12;
      case Code.historiesDeserialization:
        return 13;
      case Code.documentsDeserialization:
        return 14;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

extension IntToCode on int {
  /// Creates the error code from a discriminant.
  Code toCode() {
    switch (this) {
      case -2:
        return Code.fault;
      case -1:
        return Code.panic;
      case 0:
        return Code.none;
      case 1:
        return Code.vocabPointer;
      case 2:
        return Code.modelPointer;
      case 3:
        return Code.readFile;
      case 4:
        return Code.initAi;
      case 5:
        return Code.aiPointer;
      case 6:
        return Code.historiesPointer;
      case 7:
        return Code.historyIdPointer;
      case 8:
        return Code.documentsPointer;
      case 9:
        return Code.documentIdPointer;
      case 10:
        return Code.documentSnippetPointer;
      case 11:
        return Code.rerankerDeserialization;
      case 12:
        return Code.rerankerSerialization;
      case 13:
        return Code.historiesDeserialization;
      case 14:
        return Code.documentsDeserialization;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

@JS()
@anonymous
class XaynAiError {
  external int get code;

  external String get message;

  external factory XaynAiError({int code, String message});
}

extension XaynAiErrorToException on XaynAiError {
  /// Creates an exception from the error information.
  XaynAiException toException() => XaynAiException(code.toCode(), message);
}

@JS('WebAssembly.RuntimeError')
class RuntimeError {}

extension RuntimeErrorToException on RuntimeError {
  /// Creates an exception with a [`Code.panic`] from the JS runtime error.
  XaynAiException toException() => XaynAiException(
        Code.panic,
        'JS WebAssembly RuntimeError: $this',
      );
}
