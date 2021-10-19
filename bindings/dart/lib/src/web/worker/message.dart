import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart' show JsonSerializable;
import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Sender;
import 'package:xayn_ai_ffi_dart/src/web/worker/utils.dart'
    show Uint8ListConverter, Uint8ListNullConverter;

part 'message.g.dart';

abstract class ToJson {
  Map<String, dynamic> toJson();
}

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
  @Uint8ListNullConverter()
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
class Response {
  final Map<String, dynamic>? result;
  final XaynAiError? error;

  static Response fromResult<R extends ToJson>(R result) =>
      Response(result.toJson(), null);
  static Response fromError(XaynAiError error) => Response(null, error);
  static final ok = Response(null, null);

  Response(this.result, this.error);

  bool isError() => error != null ? true : false;

  factory Response.fromJson(Map json) => _$ResponseFromJson(json);

  Map<String, dynamic> toJson() => _$ResponseToJson(this);
}

@JsonSerializable()
class XaynAiError {
  final int code;

  final String message;

  XaynAiError(this.code, this.message);

  factory XaynAiError.fromJson(Map json) => _$XaynAiErrorFromJson(json);

  Map<String, dynamic> toJson() => _$XaynAiErrorToJson(this);
}

@JsonSerializable()
class FaultsResponse implements ToJson {
  final List<String> faults;

  FaultsResponse(
    this.faults,
  );

  factory FaultsResponse.fromJson(Map json) => _$FaultsResponseFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$FaultsResponseToJson(this);
}

@JsonSerializable()
class SerializeResponse implements ToJson {
  @Uint8ListConverter()
  final Uint8List data;

  SerializeResponse(
    this.data,
  );

  factory SerializeResponse.fromJson(Map json) =>
      _$SerializeResponseFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$SerializeResponseToJson(this);
}
