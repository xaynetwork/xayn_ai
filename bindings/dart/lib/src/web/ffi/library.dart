@JS()
library library;

import 'dart:html' show WorkerGlobalScope;
import 'package:js/js.dart' show JS;
import 'package:js/js_util.dart' show promiseToFuture;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart'
    show selectThreadPoolSize;

@JS('Promise')
class Promise<T> {}

@JS('WebAssembly.Exports')
class Wasm {}
