import 'dart:html' show MessagePort;
import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart' show JsonConverter;

abstract class ToJson {
  Map<String, dynamic> toJson();
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

class Uint8ListMaybeNullConverter
    implements JsonConverter<Uint8List?, Uint8List?> {
  const Uint8ListMaybeNullConverter();

  @override
  Uint8List? fromJson(Uint8List? json) {
    return json;
  }

  @override
  Uint8List? toJson(Uint8List? object) {
    return object;
  }
}

class MessagePortConverter
    implements JsonConverter<MessagePort?, MessagePort?> {
  const MessagePortConverter();

  @override
  MessagePort? fromJson(MessagePort? json) {
    return json;
  }

  @override
  MessagePort? toJson(MessagePort? object) {
    return object;
  }
}
