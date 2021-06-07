@JS()
library error;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show Code, IntToCode, XaynAiException;

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
