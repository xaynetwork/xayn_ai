@JS()
library error;

import 'package:js/js.dart' show JS;
import 'package:js/js_util.dart' show getProperty, hasProperty;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show Code, IntToCode, XaynAiException;

class XaynAiError extends Error {
  final int code;
  final String message;

  static bool isXaynAiError(Object o) {
    return hasProperty(o, 'code') && hasProperty(o, 'message');
  }

  XaynAiError(this.code, this.message);
}

extension ObjectToXaynAiError on Object {
  XaynAiError toXaynAiError() => XaynAiError(
        getProperty(this, 'code') as int,
        getProperty(this, 'message') as String,
      );
}

extension XaynAiErrorToException on XaynAiError {
  /// Creates an exception from the error information.
  XaynAiException toException() => XaynAiException(code.toCode(), message);
}

@JS('WebAssembly.RuntimeError')
// see: https://github.com/lexaknyazev/wasm.dart/blob/a6c93afea4732c140f1f61f144795961c42c8613/wasm_interop/lib/wasm_interop.dart#L718
external Function get runtimeError;

class RuntimeError extends Error {
  final String message;

  RuntimeError(this.message);
}

extension ObjectToRuntimeError on Object {
  RuntimeError toRuntimeError() =>
      RuntimeError(getProperty(this, 'message') as String);
}

extension RuntimeErrorToException on RuntimeError {
  /// Creates an exception with a [`Code.panic`] from the JS runtime error.
  XaynAiException toException() => XaynAiException(
        Code.panic,
        'JS WebAssembly RuntimeError: $message',
      );
}
