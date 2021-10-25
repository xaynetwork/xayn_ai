import 'dart:async' show StreamController;
import 'dart:html' show DedicatedWorkerGlobalScope;

import 'package:xayn_ai_ffi_dart/src/web/ffi/ai.dart' as ffi show XaynAi;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/request.dart'
    show Request;
import 'package:xayn_ai_ffi_dart/src/web/worker/method_handler.dart'
    show MethodHandler;

class RequestHandler<T> {
  final dws = DedicatedWorkerGlobalScope.instance;
  final _incoming = StreamController<T>();

  Stream<T> get incoming => _incoming.stream;

  RequestHandler() {
    dws.onMessage.listen((event) => _incoming.add(event.data as T));
  }
}

Future<void> handleRequests() async {
  ffi.XaynAi? ai;
  final messageHandler = RequestHandler<Map>();
  final methodHandler = MethodHandler();

  await for (final json in messageHandler.incoming) {
    final request = Request.fromJson(json);

    try {
      await methodHandler[request.method](ai, request);
    } catch (e) {
      print(e);
    }
  }
}
