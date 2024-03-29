@JS()
library ai;

import 'dart:typed_data' show Uint8List;

import 'package:js/js.dart' show JS;
import 'package:js/js_util.dart' show instanceof;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/reranker/mode.dart'
    show RerankMode, RerankModeToInt;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/web/data/document.dart'
    show JsDocument, ToJsDocuments;
import 'package:xayn_ai_ffi_dart/src/web/data/history.dart'
    show JsHistory, ToJsHistories;
import 'package:xayn_ai_ffi_dart/src/web/ffi/library.dart' show init;
import 'package:xayn_ai_ffi_dart/src/web/reranker/analytics.dart'
    show JsAnalytics, ToAnalytics;
import 'package:xayn_ai_ffi_dart/src/web/result/error.dart'
    show
        ObjectToRuntimeError,
        ObjectToXaynAiError,
        RuntimeErrorToException,
        XaynAiError,
        XaynAiErrorToException,
        runtimeError;
import 'package:xayn_ai_ffi_dart/src/web/result/fault.dart'
    show JsFault, ToStrings;
import 'package:xayn_ai_ffi_dart/src/web/result/outcomes.dart'
    show JsRerankingOutcomes, ToRerankingOutcomes;

@JS('xayn_ai_ffi_wasm.WXaynAi')
class _XaynAi {
  external _XaynAi(
    Uint8List smbertVocab,
    Uint8List smbertModel,
    Uint8List qambertVocab,
    Uint8List qambertModel,
    Uint8List ltrModel,
    Uint8List? serialized,
  );

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
class XaynAi {
  late _XaynAi? _ai;

  /// Creates and initializes the Xayn AI and initializes the WASM module.
  ///
  /// Requires the path to the vocabulary and model of the tokenizer/embedder,
  /// the path of the LTR model and the WASM module. Optionally accepts the
  /// serialized reranker database, otherwise creates a new one.
  static Future<XaynAi> create(
    Uint8List smbertVocab,
    Uint8List smbertModel,
    Uint8List qambertVocab,
    Uint8List qambertModel,
    Uint8List ltrModel,
    Uint8List wasmModule,
    Uint8List? serialized,
  ) async {
    await init(wasmModule);
    return XaynAi._(
      smbertVocab,
      smbertModel,
      qambertVocab,
      qambertModel,
      ltrModel,
      serialized,
    );
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder and the LTR model.
  /// Optionally accepts the serialized reranker database, otherwise creates a new one.
  XaynAi._(
    Uint8List smbertVocab,
    Uint8List smbertModel,
    Uint8List qambertVocab,
    Uint8List qambertModel,
    Uint8List ltrModel,
    Uint8List? serialized,
  ) {
    try {
      _ai = _XaynAi(
        smbertVocab,
        smbertModel,
        qambertVocab,
        qambertModel,
        ltrModel,
        serialized,
      );
    } catch (error) {
      if (instanceof(error, runtimeError)) {
        throw error.toRuntimeError().toException();
      } else if (XaynAiError.isXaynAiError(error)) {
        throw error.toXaynAiError().toException();
      } else {
        rethrow;
      }
    }
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  RerankingOutcomes rerank(
    RerankMode mode,
    List<History> histories,
    List<Document> documents,
  ) {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!
          .rerank(
            mode.toInt(),
            histories.toJsHistories(),
            documents.toJsDocuments(),
          )
          .toRerankingOutcomes();
    } catch (error) {
      if (instanceof(error, runtimeError)) {
        // the memory is automatically cleaned up by the js garbage collector once all reference to
        // _ai are gone, which usually happens when creating a new wasm instance
        _ai = null;
        throw error.toRuntimeError().toException();
      } else if (XaynAiError.isXaynAiError(error)) {
        throw error.toXaynAiError().toException();
      } else {
        rethrow;
      }
    }
  }

  /// Serializes the current state of the reranker.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  Uint8List serialize() {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!.serialize();
    } catch (error) {
      if (instanceof(error, runtimeError)) {
        _ai = null;
        throw error.toRuntimeError().toException();
      } else if (XaynAiError.isXaynAiError(error)) {
        throw error.toXaynAiError().toException();
      } else {
        rethrow;
      }
    }
  }

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  List<String> faults() {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!.faults().toStrings();
    } catch (error) {
      if (instanceof(error, runtimeError)) {
        _ai = null;
        throw error.toRuntimeError().toException();
      } else {
        rethrow;
      }
    }
  }

  /// Retrieves the analytics which were collected in the penultimate reranking.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  Analytics? analytics() {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!.analytics()?.toAnalytics();
    } catch (error) {
      if (instanceof(error, runtimeError)) {
        _ai = null;
        throw error.toRuntimeError().toException();
      } else {
        rethrow;
      }
    }
  }

  /// Serializes the synchronizable data of the reranker.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  Uint8List syncdataBytes() {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      return _ai!.syncdataBytes();
    } catch (error) {
      if (instanceof(error, runtimeError)) {
        _ai = null;
        throw error.toRuntimeError().toException();
      } else if (XaynAiError.isXaynAiError(error)) {
        throw error.toXaynAiError().toException();
      } else {
        rethrow;
      }
    }
  }

  /// Synchronizes the internal data of the reranker with another.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  void synchronize(Uint8List serialized) {
    if (_ai == null) {
      throw StateError('XaynAi was already freed');
    }

    try {
      _ai!.synchronize(serialized);
    } catch (error) {
      if (instanceof(error, runtimeError)) {
        _ai = null;
        throw error.toRuntimeError().toException();
      } else if (XaynAiError.isXaynAiError(error)) {
        throw error.toXaynAiError().toException();
      } else {
        rethrow;
      }
    }
  }

  /// Frees the memory.
  void free() {
    _ai?.free();
    _ai = null;
  }
}
