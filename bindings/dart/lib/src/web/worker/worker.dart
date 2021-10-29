import 'dart:html' show DedicatedWorkerGlobalScope;

import 'package:xayn_ai_ffi_dart/src/web/ffi/ai.dart' as ffi show XaynAi;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/request.dart'
    show Request;
import 'package:xayn_ai_ffi_dart/src/web/worker/method_handler.dart'
    show MethodHandler;

void main() async {
  try {
    await handleRequests();
  } catch (e) {
    print(e);
  }
}

/// A Function that handles the incoming [Request]s.
/// [Request]s are processed sequentially in the order in which they arrived.
Future<void> handleRequests() async {
  ffi.XaynAi? ai;
  final incomingMessages = DedicatedWorkerGlobalScope.instance.onMessage;
  const methodHandler = MethodHandler();

  await for (final json in incomingMessages) {
    final request = Request.fromJson(json as Map);

    try {
      await methodHandler[request.method](ai, request);
    } catch (e) {
      print(e);
    }
  }
}
