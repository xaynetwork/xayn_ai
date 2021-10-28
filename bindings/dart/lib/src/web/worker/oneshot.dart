import 'dart:html' show MessageChannel, MessageEvent, MessagePort;

import 'package:json_annotation/json_annotation.dart' show JsonSerializable;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/utils.dart'
    show MessagePortConverter;

part 'oneshot.g.dart';

class Oneshot {
  late Sender? _sender;
  late Receiver? _receiver;

  Oneshot() {
    final channel = MessageChannel();
    _sender = Sender(channel.port1);
    _receiver = Receiver(channel.port2);
  }

  Sender get sender {
    if (_sender == null) {
      throw StateError('sender was already used');
    }

    final sender = _sender!;
    _sender = null;
    return sender;
  }

  Receiver get receiver {
    if (_receiver == null) {
      throw StateError('receiver was already used');
    }

    final receiver = _receiver!;
    _receiver = null;
    return receiver;
  }
}

@JsonSerializable()
class Sender {
  @MessagePortConverter()
  late MessagePort? _port;

  Sender(this.port);

  void send(dynamic message, [List<Object>? transfer]) {
    if (port == null) {
      throw StateError('send was already called');
    }

    port!.postMessage(message, transfer);
    port!.close();
    port = null;
  }

  factory Sender.fromJson(Map json) => _$SenderFromJson(json);

  Map<String, dynamic> toJson() => _$SenderToJson(this);
}

class Receiver {
  late MessagePort? _port;

  Receiver(this._port);

  Future<MessageEvent> recv() async {
    if (_port == null) {
      throw StateError('recv was already called');
    }

    final result = await _port!.onMessage.first;
    _port!.close();
    _port = null;

    return result;
  }
}
