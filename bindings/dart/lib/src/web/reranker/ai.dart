@JS()
library ai;

import 'dart:typed_data' show Uint8List;

import 'package:js/js.dart' show JS;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' as common
    show XaynAi;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/reranker/mode.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/web/ffi/ai.dart' as ffi show XaynAi;
import 'package:xayn_ai_ffi_dart/src/web/reranker/data_provider.dart'
    show SetupData;

/// The Xayn AI.
class XaynAi implements common.XaynAi {
  final ffi.XaynAi _ai;

  /// Creates and initializes the Xayn AI from a given state.
  ///
  /// Requires the necessary [SetupData] and the state.
  /// It will throw an error if the provided state is empty.
  static Future<XaynAi> restore(SetupData data, Uint8List serialized) async {
    final ai = await ffi.XaynAi.create(
      data.smbertVocab,
      data.smbertModel,
      data.qambertVocab,
      data.qambertModel,
      data.ltrModel,
      data.wasmModule,
      serialized,
    );
    return XaynAi._(ai);
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the necessary [SetupData] for the AI.
  static Future<XaynAi> create(SetupData data) async {
    final ai = await ffi.XaynAi.create(
      data.smbertVocab,
      data.smbertModel,
      data.qambertVocab,
      data.qambertModel,
      data.ltrModel,
      data.wasmModule,
      null,
    );
    return XaynAi._(ai);
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder and the LTR model.
  /// Optionally accepts the serialized reranker database, otherwise creates a new one.
  XaynAi._(this._ai);

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<RerankingOutcomes> rerank(
    RerankMode mode,
    List<History> histories,
    List<Document> documents,
  ) async {
    return _ai.rerank(mode, histories, documents);
  }

  /// Serializes the current state of the reranker.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<Uint8List> serialize() async {
    return _ai.serialize();
  }

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<List<String>> faults() async {
    return _ai.faults();
  }

  /// Retrieves the analytics which were collected in the penultimate reranking.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<Analytics?> analytics() async {
    return _ai.analytics();
  }

  /// Serializes the synchronizable data of the reranker.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<Uint8List> syncdataBytes() async {
    return _ai.syncdataBytes();
  }

  /// Synchronizes the internal data of the reranker with another.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<void> synchronize(Uint8List serialized) async {
    _ai.synchronize(serialized);
  }

  /// Frees the memory.
  @override
  Future<void> free() async {
    _ai.free();
  }
}
