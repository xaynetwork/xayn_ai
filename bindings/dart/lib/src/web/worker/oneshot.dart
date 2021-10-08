import 'dart:html' show MessageChannel, MessageEvent, MessagePort;

import 'package:json_annotation/json_annotation.dart'
    show JsonConverter, JsonSerializable;

part 'oneshot.g.dart';

class Oneshot {
  late Sender _sender;
  late Receiver _receiver;

  Oneshot() {
    final channel = MessageChannel();
    _sender = Sender(channel.port1);
    _receiver = Receiver(channel.port2);
  }

  Sender get sender => _sender;
  Receiver get receiver => _receiver;
}

@JsonSerializable()
class Sender {
  @MessagePortConverter()
  final MessagePort port;

  Sender(this.port);

  void send(dynamic message, [List<Object>? transfer]) {
    port.postMessage(message, transfer);
    port.close();
  }

  factory Sender.fromJson(Map json) => _$SenderFromJson(json);

  Map<String, dynamic> toJson() => _$SenderToJson(this);
}

class Receiver {
  final MessagePort port;

  Receiver(this.port);

  Future<MessageEvent> recv() async {
    return await port.onMessage.first;
  }
}

class MessagePortConverter implements JsonConverter<MessagePort, MessagePort> {
  const MessagePortConverter();

  @override
  MessagePort fromJson(MessagePort json) {
    return json;
  }

  @override
  MessagePort toJson(MessagePort object) {
    return object;
  }
}
