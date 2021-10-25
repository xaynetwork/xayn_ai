import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart' show JsonSerializable;
import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/utils.dart'
    show ToJson, Uint8ListConverter, Uint8ListMaybeNullConverter;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Sender;

part 'request.g.dart';

@JsonSerializable()
class Request {
  final Method method;
  final Map<String, dynamic>? params;
  final Sender sender;

  Request(this.method, this.params, this.sender);

  factory Request.fromJson(Map json) => _$RequestFromJson(json);

  Map<String, dynamic> toJson() => _$RequestToJson(this);
}

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
  @Uint8ListMaybeNullConverter()
  Uint8List? serialized;

  final String wasmScript;

  CreateParams(this.smbertVocab, this.smbertModel, this.qambertVocab,
      this.qambertModel, this.ltrModel, this.wasmModule, this.wasmScript,
      [Uint8List? serialized]);

  factory CreateParams.fromJson(Map json) => _$CreateParamsFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$CreateParamsToJson(this);
}

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
