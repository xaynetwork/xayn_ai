import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart' show JsonSerializable;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/utils.dart' show ToJson;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/utils.dart'
    show Uint8ListConverter;

part 'response.g.dart';

/// A Response object that holds the result of the method invocation.
@JsonSerializable()
class Response implements ToJson {
  final Map<String, dynamic>? result;

  static Response fromResult<R extends ToJson>(R result) =>
      Response(result.toJson());

  static const ok = Response(null);

  const Response(this.result);

  factory Response.fromJson(Map json) => _$ResponseFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$ResponseToJson(this);
}

/// A response that contains a [Uint8List] result.
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

/// A response that holds the result of the `Method.faults` invocation.
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

/// A response that holds the result of the `Method.analytics` invocation.
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
