import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart'
    show JsonConverter, JsonSerializable;
import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Sender;

part 'message.g.dart';

@JsonSerializable()
class Message {
  final Method method;
  final Map<String, dynamic> args;
  final Sender sender;

  Message(this.method, this.args, this.sender);

  factory Message.fromJson(Map json) => _$MessageFromJson(json);

  Map<String, dynamic> toJson() => _$MessageToJson(this);
}

enum Method {
  create,
  rerank,
  faults,
  serialize,
  free,
}

@JsonSerializable()
class CreateArgs {
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
  // @Uint8ListNullConverter()
  // late Uint8List? serialized;

  final String wasmScript;

  CreateArgs(
    this.smbertVocab,
    this.smbertModel,
    this.qambertVocab,
    this.qambertModel,
    this.ltrModel,
    this.wasmModule,
    this.wasmScript,
    // [Uint8List? serialized]
  );

  factory CreateArgs.fromJson(Map json) => _$CreateArgsFromJson(json);

  Map<String, dynamic> toJson() => _$CreateArgsToJson(this);
}

class Uint8ListConverter implements JsonConverter<Uint8List, Uint8List> {
  const Uint8ListConverter();

  @override
  Uint8List fromJson(Uint8List json) {
    return json;
  }

  @override
  Uint8List toJson(Uint8List object) {
    return object;
  }
}

@JsonSerializable()
class RerankArgs {
  final RerankMode mode;
  final List<History> histories;
  final List<Document> documents;

  RerankArgs(
    this.mode,
    this.histories,
    this.documents,
  );

  factory RerankArgs.fromJson(Map json) => _$RerankArgsFromJson(json);

  Map<String, dynamic> toJson() => _$RerankArgsToJson(this);
}

@JsonSerializable()
class FaultsReturn {
  final List<String> faults;

  FaultsReturn(
    this.faults,
  );

  factory FaultsReturn.fromJson(Map json) => _$FaultsReturnFromJson(json);

  Map<String, dynamic> toJson() => _$FaultsReturnToJson(this);
}

@JsonSerializable()
class SerializeReturn {
  @Uint8ListConverter()
  final Uint8List data;

  SerializeReturn(
    this.data,
  );

  factory SerializeReturn.fromJson(Map json) => _$SerializeReturnFromJson(json);

  Map<String, dynamic> toJson() => _$SerializeReturnToJson(this);
}
