import 'dart:async' show StreamController;
import 'dart:html' show DedicatedWorkerGlobalScope;

import 'package:xayn_ai_ffi_dart/src/common/result/error.dart'
    show XaynAiException;
import 'package:xayn_ai_ffi_dart/src/web/ffi/ai.dart' as ffi show XaynAi;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/request.dart'
    show Request;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/response.dart'
    show Response;
import 'package:xayn_ai_ffi_dart/src/web/worker/method_handler.dart'
    show MethodHandler, SendResponse;

void main() async {
  try {
    await handleRequests();
  } catch (error) {
    print('Web worker error while handling a request: $error');
  }
}

/// A small wrapper around [DedicatedWorkerGlobalScope.onMessage].
/// [DedicatedWorkerGlobalScope.onMessage] does not seem to behave like a
/// real Dart [Stream]. When used in the `await for` loop in the
/// [handleRequests] function below, it sometimes loses/discards messages.
class MessageStream<T> {
  final _workerOnMessage = DedicatedWorkerGlobalScope.instance.onMessage;
  final _incoming = StreamController<T>();

  Stream<T> get incoming => _incoming.stream;

  MessageStream() {
    _workerOnMessage.listen((event) => _incoming.add(event.data as T));
  }
}

/// A function that handles the incoming [Request]s.
/// [Request]s are processed sequentially in the order in which they arrived.
Future<void> handleRequests() async {
  final messageStream = MessageStream<Map>();
  const methodHandler = MethodHandler();
  ffi.XaynAi? ai;

  await for (final json in messageStream.incoming) {
    final request = Request.fromJson(json);
    try {
      ai = await methodHandler[request.method](ai, request);
    } on XaynAiException catch (error) {
      request.sender.sendResponse(Response.fromError(error));
    } catch (error) {
      print(
          'Web worker error while handling the method call `${request.method}`: $error');
    }
  }
}
