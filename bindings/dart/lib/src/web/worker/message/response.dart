import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart' show JsonSerializable;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/utils.dart'
    show ToJson, Uint8ListConverter;

part 'response.g.dart';

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
class Uint8ListResponse implements ToJson {
  @Uint8ListConverter()
  final Uint8List data;

  Uint8ListResponse(
    this.data,
  );

  factory Uint8ListResponse.fromJson(Map json) =>
      _$Uint8ListResponseFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$Uint8ListResponseToJson(this);
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
class AnalyticsResponse implements ToJson {
  Analytics? analytics;

  AnalyticsResponse(
    this.analytics,
  );

  factory AnalyticsResponse.fromJson(Map json) =>
      _$AnalyticsResponseFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$AnalyticsResponseToJson(this);
}
