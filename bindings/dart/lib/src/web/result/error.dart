@JS()
library error;

import 'package:js/js.dart' show JS;
import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show Code, XaynAiException;

@JS('WebAssembly.RuntimeError')
class JsRuntimeException {}

extension ToXaynAiException on JsRuntimeException {
  XaynAiException toXaynAiException() => XaynAiException(
        Code.panic,
        'JS WebAssembly RuntimeError: $this',
      );
}
