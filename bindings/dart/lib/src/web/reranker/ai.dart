@JS()
library ai;

import 'dart:typed_data' show Uint32List, Uint8List;

import 'package:js/js.dart' show JS;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' as common
    show XaynAi;
import 'package:xayn_ai_ffi_dart/src/web/data/document.dart'
    show JsDocument, ToJsDocuments;
import 'package:xayn_ai_ffi_dart/src/web/data/history.dart'
    show JsHistory, ToJsHistories;
import 'package:xayn_ai_ffi_dart/src/web/reranker/analytics.dart'
    show JsAnalytics, ToAnalytics;
import 'package:xayn_ai_ffi_dart/src/web/reranker/data_provider.dart'
    show SetupData;
import 'package:xayn_ai_ffi_dart/src/web/result/error.dart'
    show
        RuntimeError,
        RuntimeErrorToException,
        XaynAiError,
        XaynAiErrorToException;
import 'package:xayn_ai_ffi_dart/src/web/result/fault.dart'
    show JsFault, ToStrings;

@JS('xayn_ai_ffi_wasm.WXaynAi')
class _XaynAi {
  external _XaynAi(Uint8List vocab, Uint8List model, [Uint8List? serialized]);

  external Uint32List rerank(
    List<JsHistory> histories,
    List<JsDocument> documents,
  );

  external Uint8List serialize();

  external List<JsFault> faults();

  external JsAnalytics? analytics();

  external void free();
}

/// The Xayn AI.
class XaynAi implements common.XaynAi {
  late _XaynAi? _ai;

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder. Optionally accepts the serialized
  /// reranker database, otherwise creates a new one.
  XaynAi(SetupData data, [Uint8List? serialized]) {
    try {
      _ai = _XaynAi(data.vocab, data.model, serialized);
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
  @override
  List<int> rerank(List<History> histories, List<Document> documents) {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!
          .rerank(histories.toJsHistories(), documents.toJsDocuments())
          .toList(growable: false);
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
  @override
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
  @override
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
  @override
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

  /// Frees the memory.
  @override
  void free() {
    _ai?.free();
    _ai = null;
  }
}