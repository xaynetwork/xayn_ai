import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' as common
    show XaynAi;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/web/reranker/data_provider.dart'
    show SetupData;
import 'package:xayn_ai_ffi_dart/src/web/worker/proxy.dart' show XaynAiWorker;

/// The Xayn AI.
class XaynAi implements common.XaynAi {
  final XaynAiWorker _ai;

  /// Creates and initializes the Xayn AI and initializes the WASM module.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder and the WASM
  /// module. Optionally accepts the serialized reranker database, otherwise
  /// creates a new one.
  static Future<XaynAi> create(SetupData data, [Uint8List? serialized]) async {
    final ai = await XaynAiWorker.create(data, serialized);
    return XaynAi._(ai);
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder. Optionally accepts the serialized
  /// reranker database, otherwise creates a new one.
  XaynAi._(this._ai);

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  @override
  Future<RerankingOutcomes> rerank(RerankMode mode, List<History> histories,
      List<Document> documents) async {
    return await _ai.rerank(mode, histories, documents);
  }

  /// Serializes the current state of the reranker.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  @override
  Future<Uint8List> serialize() async {
    return await _ai.serialize();
  }

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  @override
  Future<List<String>> faults() async {
    return await _ai.faults();
  }

  /// Retrieves the analytics which were collected in the penultimate reranking.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  @override
  Future<Analytics?> analytics() async {
    return await _ai.analytics();
  }

  /// Serializes the synchronizable data of the reranker.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  @override
  Future<Uint8List> syncdataBytes() async {
    return await _ai.syncdataBytes();
  }

  /// Synchronizes the internal data of the reranker with another.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  @override
  Future<void> synchronize(Uint8List serialized) async {
    await _ai.synchronize(serialized);
  }

  /// Frees the memory.
  @override
  Future<void> free() async {
    await _ai.free();
  }
}
