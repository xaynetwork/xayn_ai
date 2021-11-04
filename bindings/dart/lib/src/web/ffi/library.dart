@JS()
library library;

import 'dart:html' show WorkerGlobalScope;

import 'package:js/js.dart' show JS;
import 'package:js/js_util.dart' show promiseToFuture;

import 'package:xayn_ai_ffi_dart/src/common/reranker/utils.dart'
    show selectThreadPoolSize;

@JS('Promise')
class _Promise<T> {}

@JS('WebAssembly.Exports')
class Wasm {}

@JS('xayn_ai_ffi_wasm.default')
external _Promise<Wasm> _init([
  // ignore: non_constant_identifier_names
  dynamic module_or_path,
]);

@JS('xayn_ai_ffi_wasm.initThreadPool')
external _Promise<void> _initThreadPool(int numberOfThreads);

/// Initializes the wasm module.
///
/// If `moduleOrPath` is a `RequestInfo` or `URL`, makes a request and
/// for everything else, calls `WebAssembly.instantiate` directly.
Future<Wasm> init([dynamic moduleOrPath]) async {
  final wasm = await promiseToFuture<Wasm>(_init(moduleOrPath));

  // Most devices have 4+ hardware threads, but if the browser doesn't support
  // the property it's probably old so we default to 2.
  var hardwareThreads = selectThreadPoolSize(
      WorkerGlobalScope.instance.navigator.hardwareConcurrency ?? 2);

  await promiseToFuture<void>(_initThreadPool(hardwareThreads));
  return wasm;
}
