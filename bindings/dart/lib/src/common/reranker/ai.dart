import 'dart:math' show max, min;
import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart' show JsonValue;
import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/ffi/genesis.dart' as ffi
    show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    show SetupData;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;

/// Rerank mode
enum RerankMode {
  @JsonValue(ffi.RerankMode.StandardNews)
  standardNews,
  @JsonValue(ffi.RerankMode.PersonalizedNews)
  personalizedNews,
  @JsonValue(ffi.RerankMode.StandardSearch)
  standardSearch,
  @JsonValue(ffi.RerankMode.PersonalizedSearch)
  personalizedSearch,
}

extension RerankModeToInt on RerankMode {
  /// Gets the discriminant.
  int toInt() {
    // We can't use `_$RerankModeEnumMap` as it only gets generated for
    // files which have a `@JsonSerializable` type containing the enum.
    // You can't make enums `@JsonSerializable`. Given that `RerankMode`
    // has only few variants and rarely changes we just write this switch
    // statement by hand.
    switch (this) {
      case RerankMode.standardNews:
        return ffi.RerankMode.StandardNews;
      case RerankMode.personalizedNews:
        return ffi.RerankMode.PersonalizedNews;
      case RerankMode.standardSearch:
        return ffi.RerankMode.StandardSearch;
      case RerankMode.personalizedSearch:
        return ffi.RerankMode.PersonalizedSearch;
    }
  }
}

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

/// Maximum number of threads to be used for multithreaded features.
const int maxNumberOfThreads = 16;

/// Selects the number of threads used by the [`XaynAi`] thread pool.
///
/// On a single core system the thread pool consists of only one thread.
/// On a multicore system the thread pool consists of
/// (the number of logical cores - 1) threads, but at most [`maxNumberOfThreads`]
/// threads and at least one thread.
int selectThreadPoolSize(int numberOfProcessors) =>
    min(max(numberOfProcessors - 1, 1), maxNumberOfThreads);
