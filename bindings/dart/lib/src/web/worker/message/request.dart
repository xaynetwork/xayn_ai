import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart' show JsonSerializable;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/mode.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/utils.dart' show ToJson;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/utils.dart'
    show Uint8ListConverter, Uint8ListMaybeNullConverter;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Sender;

part 'request.g.dart';

/// The name of the method to be invoked.
enum Method {
  create,
  rerank,
  faults,
  serialize,
  analytics,
  syncdataBytes,
  synchronize,
  free,
}

/// A request object that holds the following fields:
/// - the [Method] to be invoked,
/// - the parameters used during the invocation of the [Method] (it may be omitted),
/// - the [Sender] used by the producer to send back the result.
@JsonSerializable()
class Request implements ToJson {
  final Method method;
  final Map<String, dynamic>? params;
  final Sender sender;

  Request(this.method, this.params, this.sender);

  factory Request.fromJson(Map json) => _$RequestFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$RequestToJson(this);
}

/// The parameters to be used during the invocation of the [Method.create] method.
@JsonSerializable()
class CreateParams implements ToJson {
  @Uint8ListConverter()
  final Uint8List smbertVocab;
  @Uint8ListConverter()
  final Uint8List smbertModel;
  @Uint8ListConverter()
  final Uint8List qambertVocab;
  @Uint8ListConverter()
  final Uint8List qambertModel;
  @Uint8ListConverter()
  final Uint8List ltrModel;
  @Uint8ListConverter()
  final Uint8List wasmModule;
  final String wasmScript;
  @Uint8ListMaybeNullConverter()
  final Uint8List? serialized;

  CreateParams(
    this.smbertVocab,
    this.smbertModel,
    this.qambertVocab,
    this.qambertModel,
    this.ltrModel,
    this.wasmModule,
    this.wasmScript,
    this.serialized,
  );

  factory CreateParams.fromJson(Map json) => _$CreateParamsFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$CreateParamsToJson(this);
}

/// The parameters to be used during the invocation of the [Method.rerank] method.
@JsonSerializable()
class RerankParams implements ToJson {
  final RerankMode mode;
  final List<History> histories;
  final List<Document> documents;

  RerankParams(
    this.mode,
    this.histories,
    this.documents,
  );

  factory RerankParams.fromJson(Map json) => _$RerankParamsFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$RerankParamsToJson(this);
}

/// The parameters to be used during the invocation of the [Method.synchronize] method.
@JsonSerializable()
class SynchronizeParams implements ToJson {
  @Uint8ListConverter()
  final Uint8List serialized;

  SynchronizeParams(
    this.serialized,
  );

  factory SynchronizeParams.fromJson(Map json) =>
      _$SynchronizeParamsFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$SynchronizeParamsToJson(this);
}
