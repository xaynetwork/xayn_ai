import 'dart:html' show MessageChannel, MessageEvent, MessagePort;

import 'package:json_annotation/json_annotation.dart' show JsonSerializable;
import 'package:xayn_ai_ffi_dart/src/common/utils.dart' show ToJson;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/utils.dart'
    show MessagePortConverter;

part 'oneshot.g.dart';

/// A oneshot channel for sending a single message between different browsing
/// contexts (e.g. main thread to worker/worker to worker).
class Oneshot {
  late Sender? _sender;
  late Receiver? _receiver;

  /// Creates a new oneshot channel with a [Sender] and [Receiver] handle.
  Oneshot() {
    final channel = MessageChannel();
    _sender = Sender(channel.port1);
    _receiver = Receiver(channel.port2);
  }

  /// Returns the [Sender] handle.
  ///
  /// The method can only be called once. Calling the [Oneshot.sender]
  /// method again leads to a [StateError].
  Sender get sender {
    if (_sender == null) {
      throw StateError('sender was already used');
    }

    final sender = _sender!;
    _sender = null;
    return sender;
  }

  /// Returns the [Receiver] handle.
  ///
  /// The method can only be called once. Calling the [Oneshot.receiver]
  /// method again leads to a [StateError].
  Receiver get receiver {
    if (_receiver == null) {
      throw StateError('receiver was already used');
    }

    final receiver = _receiver!;
    _receiver = null;
    return receiver;
  }
}

/// A sender handle used by the producer to send a message.
@JsonSerializable()
class Sender implements ToJson {
  @MessagePortConverter()
  late MessagePort? port;

  // Creates a new sender handle.
  Sender(this.port);

  /// Sends a message to the [Receiver].
  ///
  /// The method can only be called once. Calling the [Sender.send]
  /// method again leads to a [StateError].
  ///
  /// If the [Receiver] half closes the channel, calling [Sender.send]
  /// will silently fail.
  void send(dynamic message, [List<Object>? transfer]) {
    if (port == null) {
      throw StateError('Sender.send was already called');
    }

    port!.postMessage(message, transfer);
    port!.close();
    port = null;
  }

  factory Sender.fromJson(Map json) => _$SenderFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$SenderToJson(this);
}

/// A receiver handle used by the consumer to receive a [MessageEvent].
class Receiver {
  late MessagePort? _port;

  // Creates a new receiver handle.
  Receiver(this._port);

  /// Waits until the [MessageEvent] has been received.
  ///
  /// The method can only be called once. Calling the [Receiver.recv]
  /// method again leads to a [StateError].
  ///
  /// If the [Sender] half closes the channel without sending an event,
  /// [Receiver.recv] will never complete.
  Future<MessageEvent> recv() async {
    if (_port == null) {
      throw StateError('Receiver.recv was already called');
    }

    final result = await _port!.onMessage.first;
    _port!.close();
    _port = null;

    return result;
  }
}
