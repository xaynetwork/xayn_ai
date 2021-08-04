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
  @JsonValue(ffi.RerankMode.News)
  news,
  @JsonValue(ffi.RerankMode.Search)
  search,
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
      case RerankMode.news:
        return ffi.RerankMode.News;
      case RerankMode.search:
        return ffi.RerankMode.Search;
    }
  }
}

/// The Xayn AI.
class XaynAi {
  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the data to setup the AI, e.g. the vocabulary and model of the tokenizer/embedder.
  /// Optionally accepts the serialized reranker database, otherwise creates a new one.
  static Future<XaynAi> create(SetupData data, [Uint8List? serialized]) async {
    throw UnsupportedError('Unsupported platform.');
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  RerankingOutcomes rerank(
          RerankMode mode, List<History> histories, List<Document> documents) =>
      throw UnsupportedError('Unsupported platform.');

  /// Serializes the current state of the reranker.
  Uint8List serialize() => throw UnsupportedError('Unsupported platform.');

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  List<String> faults() => throw UnsupportedError('Unsupported platform.');

  /// Retrieves the analytics which were collected in the penultimate reranking.
  Analytics? analytics() => throw UnsupportedError('Unsupported platform.');

  /// Serializes the synchronizable data of the reranker.
  Uint8List syncdataBytes() => throw UnsupportedError('Unsupported platform.');

  /// Synchronizes the internal data of the reranker with another.
  void synchronize([Uint8List? serialized]) =>
      throw UnsupportedError('Unsupported platform');

  /// Frees the memory.
  void free() => throw UnsupportedError('Unsupported platform.');
}
