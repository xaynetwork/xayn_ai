import 'dart:async' show StreamController;
import 'dart:html' show DedicatedWorkerGlobalScope, WorkerGlobalScope;

import 'package:xayn_ai_ffi_dart/src/web/ffi/library.dart';
import 'package:xayn_ai_ffi_dart/src/web/worker/message.dart'
    show CreateArgs, FaultsReturn, Message, Method, RerankArgs, SerializeReturn;

class MessageHandler<T> {
  final dws = DedicatedWorkerGlobalScope.instance;
  final _incoming = StreamController<T>.broadcast();
  final _outgoing = StreamController<T>.broadcast();

  Stream<T> get onMessage => _incoming.stream;
  Sink<T> get postMessage => _outgoing.sink;

  MessageHandler() {
    dws.onMessage.listen((event) => _incoming.add(event.data as T));
    _outgoing.stream.listen(dws.postMessage);
  }
}

void main() async {
  try {
    final messageHandler = MessageHandler<Map>();
    JSXaynAi? ai;

    messageHandler.onMessage.listen((Map map) async {
      final message = Message.fromJson(map);

      if (message.method == Method.create) {
        ai = await create(message);
      }

      if (message.method == Method.rerank) {
        rerank(ai!, message);
      }

      if (message.method == Method.faults) {
        final result = ai!.faults();
        message.sender.send({
          'return': FaultsReturn(result).toJson(),
        });
      }

      if (message.method == Method.serialize) {
        final result = ai!.serialize();
        message.sender.send({
          'return': SerializeReturn(result).toJson(),
        });
      }

      if (message.method == Method.free) {
        ai!.free();
        ai = null;
        message.sender.send({
          'return': null,
        });
      }
    });
  } catch (e) {
    print(e);
  }
}

Future<JSXaynAi> create(Message message) async {
  final args = CreateArgs.fromJson(message.args);
  WorkerGlobalScope.instance.importScripts(args.wasmScript);

  final ai = await JSXaynAi.create(args.smbertVocab, args.smbertModel,
      args.qambertVocab, args.qambertModel, args.ltrModel, args.wasmModule);

  message.sender.send({
    'return': null,
  });

  return ai;
}

void rerank(JSXaynAi ai, Message message) {
  final args = RerankArgs.fromJson(message.args);
  final result = ai.rerank(args.mode, args.histories, args.documents);

  message.sender.send({
    'return': result.toJson(),
  });
}
