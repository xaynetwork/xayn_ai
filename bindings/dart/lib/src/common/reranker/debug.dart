import 'dart:convert' show base64Decode, base64Encode, jsonDecode, jsonEncode;
import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart'
    show JsonSerializable, JsonKey;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;

part 'debug.g.dart';

/// Bundle of all reranking call data used for debugging purpose.
///
/// This combines the data used for a reranking call including
/// the documents, history and (optionally) the serialized state.
///
/// It provides a to/from json serialization and is meant for Team Blue
/// so that they can store JSON blobs of the Reranking call data for
/// debugging purpose.
///
/// Furthermore both the dart example/benchmark app and the dev-tool
/// provide ways to run reranking based on the serialized call data
/// and can be used for debugging.
///
/// To make this possible it was made sure that the (JSON) serialization
/// format between dart (native,js) and rust is the same. This also
/// means that all fields are renamed to snake-case. E.g. `serializedState`
/// gets encoded as `serialized_state`.
///
@JsonSerializable()
class RerankDebugCallData {
  /// The mode which was used to run the reranking.
  final RerankMode rerankMode;

  /// History used for a reranking call.
  final List<History> histories;

  /// Documents used for a reranking call.
  final List<Document> documents;

  /// Serialized state which should be used to run a reranking call.
  ///
  /// This is normally the state *before* histories/documents
  /// were used for a reranking call.
  ///
  /// This field is JSON encoded as base64 encoded string.
  @JsonKey(toJson: _optBytesToBase64, fromJson: _optBase64ToBytes)
  final Uint8List? serializedState;

  /// Creates a new instance.
  RerankDebugCallData({
    required this.rerankMode,
    required this.histories,
    required this.documents,
    this.serializedState,
  });

  /// Creates an instance from a JSON map.
  factory RerankDebugCallData.fromJson(Map<String, dynamic> json) =>
      _$RerankDebugCallDataFromJson(json);

  /// Creates an instance from a JSON String.
  factory RerankDebugCallData.fromJsonString(String json) =>
      RerankDebugCallData.fromJson(jsonDecode(json) as Map<String, dynamic>);

  /// Creates a JSON map based on this instance.
  ///
  /// Serialized state is included as a base64 encoded string.
  Map<String, dynamic> toJson() => _$RerankDebugCallDataToJson(this);

  /// Creates a JSON string based on this instance.
  String toJsonString() => jsonEncode(toJson());
}

String? _optBytesToBase64(Uint8List? bytes) =>
    bytes == null ? null : base64Encode(bytes);

Uint8List? _optBase64ToBytes(String? base64) =>
    base64 == null ? null : base64Decode(base64);
