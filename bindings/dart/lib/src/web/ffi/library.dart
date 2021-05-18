@JS()
library library;

import 'package:js/js.dart' show JS;
import 'package:js/js_util.dart' show promiseToFuture;

@JS('Promise')
class _Promise<T> {}

@JS('WebAssembly.Exports')
class Wasm {}

@JS('xayn_ai_ffi_wasm.default')
external _Promise<Wasm> _init([
  // ignore: non_constant_identifier_names
  dynamic? module_or_path,
]);

/// Initializes the wasm module.
///
/// If `moduleOrPath` is a `RequestInfo` or `URL`, makes a request and
/// for everything else, calls `WebAssembly.instantiate` directly.
Future<Wasm> init([dynamic? moduleOrPath]) async =>
    promiseToFuture(_init(moduleOrPath));
