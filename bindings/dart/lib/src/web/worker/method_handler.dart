import 'dart:html' show WorkerGlobalScope;

import 'package:xayn_ai_ffi_dart/src/web/ffi/ai.dart' as ffi show XaynAi;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/request.dart'
    show CreateParams, Method, Request, RerankParams, SynchronizeParams;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/response.dart'
    show AnalyticsResponse, FaultsResponse, Response, Uint8ListResponse;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Sender;

typedef Handler = Future<void> Function(ffi.XaynAi? ai, Request request);

class MethodHandler {
  final handlers = <Method, Handler>{
    Method.create: create,
    Method.rerank: rerank,
    Method.faults: faults,
    Method.serialize: serialize,
    Method.analytics: analytics,
    Method.syncdataBytes: syncdataBytes,
    Method.synchronize: synchronize,
    Method.free: free,
  };

  Handler operator [](Method method) {
    try {
      return handlers[method]!;
    } catch (_) {
      throw NoSuchMethodError;
    }
  }
}

Future<void> create(ffi.XaynAi? ai, Request request) async {
  final params = CreateParams.fromJson(request.params!);
  WorkerGlobalScope.instance.importScripts(params.wasmScript);

  ai = await ffi.XaynAi.create(
      params.smbertVocab,
      params.smbertModel,
      params.qambertVocab,
      params.qambertModel,
      params.ltrModel,
      params.wasmModule,);

  send(request.sender, Response.ok);
}

Future<void> rerank(ffi.XaynAi? ai, Request request) async {
  final params = RerankParams.fromJson(request.params!);
  final result = ai!.rerank(params.mode, params.histories, params.documents);
  send(request.sender, Response.fromResult(result));
}

Future<void> faults(ffi.XaynAi? ai, Request request) async {
  final result = ai!.faults();
  send(request.sender, Response.fromResult(FaultsResponse(result)));
}

Future<void> serialize(ffi.XaynAi? ai, Request request) async {
  final result = ai!.serialize();
  send(request.sender, Response.fromResult(Uint8ListResponse(result)));
}

Future<void> analytics(ffi.XaynAi? ai, Request request) async {
  final result = ai!.analytics();
  send(request.sender, Response.fromResult(AnalyticsResponse(result)));
}

Future<void> syncdataBytes(ffi.XaynAi? ai, Request request) async {
  final result = ai!.syncdataBytes();
  send(request.sender, Response.fromResult(Uint8ListResponse(result)));
}

Future<void> synchronize(ffi.XaynAi? ai, Request request) async {
  final params = SynchronizeParams.fromJson(request.params!);
  ai!.synchronize(params.serialized);
  send(request.sender, Response.ok);
}

Future<void> free(ffi.XaynAi? ai, Request request) async {
  ai!.free();
  ai = null;
  send(request.sender, Response.ok);
}

void send(Sender sender, Response response) async {
  sender.send(response.toJson());
}
