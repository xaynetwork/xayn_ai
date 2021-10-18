import 'dart:async' show StreamController;
import 'dart:html' show DedicatedWorkerGlobalScope, WorkerGlobalScope;

import 'package:xayn_ai_ffi_dart/src/web/ffi/library.dart' show JSXaynAi;
import 'package:xayn_ai_ffi_dart/src/web/worker/message.dart'
    show
        CreateParams,
        FaultsResponse,
        Method,
        Request,
        RerankParams,
        Response,
        SerializeResponse,
        XaynAiError;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Sender;

class MessageHandler<T> {
  final dws = DedicatedWorkerGlobalScope.instance;
  final _incoming = StreamController<T>();

  Stream<T> get incoming => _incoming.stream;

  MessageHandler() {
    dws.onMessage.listen((event) => _incoming.add(event.data as T));
  }
}

void main() async {
  final messageHandler = MessageHandler<Map>();
  JSXaynAi? ai;

  await for (final json in messageHandler.incoming) {
    try {
      final request = Request.fromJson(json);

      try {
        switch (request.method) {
          case Method.create:
            ai = await create(request);
            break;
          case Method.rerank:
            rerank(ai!, request);
            break;
          case Method.faults:
            faults(ai!, request);
            break;
          case Method.serialize:
            serialize(ai!, request);
            break;
          case Method.free:
            ai!.free();
            ai = null;
            send(request.sender, Response.ok);
            break;
          default:
            throw UnsupportedError('Undefined enum variant.');
        }
      } on XaynAiError catch (error) {
        send(request.sender, Response.fromError(error));
      } catch (other) {
        send(request.sender,
            Response.fromError(XaynAiError(0, other.toString())));
      }
    } catch (e) {
      print(e);
    }
  }
}

Future<JSXaynAi> create(Request request) async {
  final params = CreateParams.fromJson(request.params!);
  WorkerGlobalScope.instance.importScripts(params.wasmScript);

  final ai = await JSXaynAi.create(
      params.smbertVocab,
      params.smbertModel,
      params.qambertVocab,
      params.qambertModel,
      params.ltrModel,
      params.wasmModule);

  send(request.sender, Response.ok);

  return ai;
}

void rerank(JSXaynAi ai, Request request) {
  final params = RerankParams.fromJson(request.params!);
  final result = ai.rerank(params.mode, params.histories, params.documents);
  send(request.sender, Response.fromResult(result.toJson()));
}

void faults(JSXaynAi ai, Request request) {
  final result = ai.faults();
  send(request.sender, Response.fromResult(FaultsResponse(result).toJson()));
}

void serialize(JSXaynAi ai, Request request) {
  final result = ai.serialize();
  send(request.sender, Response.fromResult(SerializeResponse(result).toJson()));
}

void send(Sender sender, Response response) {
  sender.send(response.toJson());
}
