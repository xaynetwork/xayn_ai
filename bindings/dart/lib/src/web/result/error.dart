@JS()
library error;

import 'package:js/js.dart' show JS;

@JS('WebAssembly.RuntimeError')
class JsRuntimeException {}
