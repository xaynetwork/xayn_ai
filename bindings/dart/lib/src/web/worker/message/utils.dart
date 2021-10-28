import 'dart:html' show MessagePort;
import 'dart:typed_data' show Uint8List;

import 'package:json_annotation/json_annotation.dart' show JsonConverter;

/// A [Uint8List] from/to JSON converter.
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

/// A `Uint8List?` from/to JSON converter.
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

/// A `MessagePort?` from/to JSON converter.
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
