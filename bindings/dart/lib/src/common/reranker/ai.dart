import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    show SetupData;
import 'package:xayn_ai_ffi_dart/src/common/reranker/mode.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;

/// The Xayn AI.
class XaynAi {
  /// Creates and initializes the Xayn AI from a given state.
  ///
  /// Requires the necessary [SetupData] and the state.
  /// It will throw an error if the provided state is empty.
  static Future<XaynAi> restore(SetupData data, Uint8List serialized) async {
    throw UnsupportedError('Unsupported platform.');
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the necessary [SetupData] for the AI.
  static Future<XaynAi> create(SetupData data) async {
    throw UnsupportedError('Unsupported platform.');
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  Future<RerankingOutcomes> rerank(
    RerankMode mode,
    List<History> histories,
    List<Document> documents,
  ) async =>
      throw UnsupportedError('Unsupported platform.');

  /// Serializes the current state of the reranker.
  Future<Uint8List> serialize() async =>
      throw UnsupportedError('Unsupported platform.');

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  Future<List<String>> faults() async =>
      throw UnsupportedError('Unsupported platform.');

  /// Retrieves the analytics which were collected in the penultimate reranking.
  Future<Analytics?> analytics() async =>
      throw UnsupportedError('Unsupported platform.');

  /// Serializes the synchronizable data of the reranker.
  Future<Uint8List> syncdataBytes() async =>
      throw UnsupportedError('Unsupported platform.');

  /// Synchronizes the internal data of the reranker with another.
  Future<void> synchronize(Uint8List serialized) async =>
      throw UnsupportedError('Unsupported platform');

  /// Frees the memory.
  Future<void> free() async => throw UnsupportedError('Unsupported platform.');
}
