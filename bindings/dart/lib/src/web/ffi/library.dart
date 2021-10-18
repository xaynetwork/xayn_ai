@JS()
library library;

import 'dart:html' show WorkerGlobalScope;
import 'dart:typed_data' show Uint8List;

import 'package:js/js.dart' show JS;
import 'package:js/js_util.dart' show promiseToFuture;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart'
    show RerankMode, RerankModeToInt, selectThreadPoolSize;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/web/data/document.dart'
    show JsDocument, ToJsDocuments;
import 'package:xayn_ai_ffi_dart/src/web/data/history.dart'
    show JsHistory, ToJsHistories;
import 'package:xayn_ai_ffi_dart/src/web/reranker/analytics.dart'
    show JsAnalytics, ToAnalytics;
import 'package:xayn_ai_ffi_dart/src/web/result/error.dart'
    show
        RuntimeError,
        RuntimeErrorToException,
        XaynAiError,
        XaynAiErrorToException;
import 'package:xayn_ai_ffi_dart/src/web/result/fault.dart'
    show JsFault, ToStrings;
import 'package:xayn_ai_ffi_dart/src/web/result/outcomes.dart'
    show JsRerankingOutcomes, ToRerankingOutcomes;

@JS('Promise')
class Promise<T> {}

@JS('WebAssembly.Exports')
class Wasm {}

// in genesis.js we need to export to self.wasm_bindgen = wasm_bindgen
@JS('wasm_bindgen')
external Promise<Wasm> _wasmBindgen([
  // ignore: non_constant_identifier_names
  dynamic module_or_path,
]);

@JS('wasm_bindgen.initThreadPool')
external Promise<void> _initThreadPool(int numberOfThreads);

/// Initializes the wasm module.
///
/// If `moduleOrPath` is a `RequestInfo` or `URL`, makes a request and
/// for everything else, calls `WebAssembly.instantiate` directly.
Future<Wasm> init([dynamic moduleOrPath]) async {
  final wasm = await promiseToFuture<Wasm>(_wasmBindgen(moduleOrPath));

  // Most devices have 4+ hardware threads, but if the browser doesn't support
  // the property it's probably old so we default to 2.
  var hardwareThreads = selectThreadPoolSize(
      WorkerGlobalScope.instance.navigator.hardwareConcurrency ?? 2);

  await promiseToFuture<void>(_initThreadPool(hardwareThreads));
  return wasm;
}

@JS('wasm_bindgen.WXaynAi')
class _XaynAi {
  external _XaynAi(Uint8List smbertVocab, Uint8List smbertModel,
      Uint8List qambertVocab, Uint8List qambertModel, Uint8List ltrModel,
      [Uint8List? serialized]);

  external JsRerankingOutcomes rerank(
    int mode,
    List<JsHistory> histories,
    List<JsDocument> documents,
  );

  external Uint8List serialize();

  external List<JsFault> faults();

  external JsAnalytics? analytics();

  external Uint8List syncdataBytes();

  external void synchronize(Uint8List serialized);

  external void free();
}

/// The Xayn AI.
class JSXaynAi {
  late _XaynAi? _ai;

  /// Creates and initializes the Xayn AI and initializes the WASM module.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder and the WASM
  /// module. Optionally accepts the serialized reranker database, otherwise
  /// creates a new one.
  static Future<JSXaynAi> create(
      Uint8List smbertVocab,
      Uint8List smbertModel,
      Uint8List qambertVocab,
      Uint8List qambertModel,
      Uint8List ltrModel,
      Uint8List wasmModule,
      [Uint8List? serialized]) async {
    await init(wasmModule);
    return JSXaynAi._(smbertVocab, smbertModel, qambertVocab, qambertModel,
        ltrModel, serialized);
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder. Optionally accepts the serialized
  /// reranker database, otherwise creates a new one.
  JSXaynAi._(Uint8List smbertVocab, Uint8List smbertModel,
      Uint8List qambertVocab, Uint8List qambertModel, Uint8List ltrModel,
      [Uint8List? serialized]) {
    try {
      _ai = _XaynAi(smbertVocab, smbertModel, qambertVocab, qambertModel,
          ltrModel, serialized);
    } on XaynAiError catch (error) {
      throw error.toException();
    } on RuntimeError catch (error) {
      throw error.toException();
    }
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  RerankingOutcomes rerank(
      RerankMode mode, List<History> histories, List<Document> documents) {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!
          .rerank(mode.toInt(), histories.toJsHistories(),
              documents.toJsDocuments())
          .toRerankingOutcomes();
    } on XaynAiError catch (error) {
      throw error.toException();
    } on RuntimeError catch (error) {
      // the memory is automatically cleaned up by the js garbage collector once all reference to
      // _ai are gone, which usually happens when creating a new wasm instance
      _ai = null;
      throw error.toException();
    }
  }

  /// Serializes the current state of the reranker.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  Uint8List serialize() {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!.serialize();
    } on XaynAiError catch (error) {
      throw error.toException();
    } on RuntimeError catch (error) {
      _ai = null;
      throw error.toException();
    }
  }

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  List<String> faults() {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!.faults().toStrings();
    } on RuntimeError catch (error) {
      _ai = null;
      throw error.toException();
    }
  }

  /// Retrieves the analytics which were collected in the penultimate reranking.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  Analytics? analytics() {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!.analytics()?.toAnalytics();
    } on RuntimeError catch (error) {
      _ai = null;
      throw error.toException();
    }
  }

  /// Serializes the synchronizable data of the reranker.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  Uint8List syncdataBytes() {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!.syncdataBytes();
    } on XaynAiError catch (error) {
      throw error.toException();
    } on RuntimeError catch (error) {
      _ai = null;
      throw error.toException();
    }
  }

  /// Synchronizes the internal data of the reranker with another.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  void synchronize(Uint8List serialized) {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      _ai!.synchronize(serialized);
    } on XaynAiError catch (error) {
      throw error.toException();
    } on RuntimeError catch (error) {
      _ai = null;
      throw error.toException();
    }
  }

  /// Frees the memory.
  void free() {
    _ai?.free();
    _ai = null;
  }
}
