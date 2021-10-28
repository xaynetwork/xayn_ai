import 'dart:html' show WorkerGlobalScope;

import 'package:xayn_ai_ffi_dart/src/web/ffi/ai.dart' as ffi show XaynAi;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/request.dart'
    show CreateParams, Method, Request, RerankParams, SynchronizeParams;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/response.dart'
    show AnalyticsResponse, FaultsResponse, Response, Uint8ListResponse;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Sender;

typedef Handler = Future<void> Function(ffi.XaynAi? ai, Request request);

/// A method handler for handling [Method] invocations.
class MethodHandler {
  static const handlers = <Method, Handler>{
    Method.create: create,
    Method.rerank: rerank,
    Method.faults: faults,
    Method.serialize: serialize,
    Method.analytics: analytics,
    Method.syncdataBytes: syncdataBytes,
    Method.synchronize: synchronize,
    Method.free: free,
  };

  const MethodHandler();

  Handler operator [](Method method) {
    try {
      return handlers[method]!;
    } catch (_) {
      throw NoSuchMethodError;
    }
  }
}

/// The method handler for the [Method.create] invocation.
Future<void> create(ffi.XaynAi? ai, Request request) async {
  final params = CreateParams.fromJson(request.params!);
  WorkerGlobalScope.instance.importScripts(params.wasmScript);

  ai = await ffi.XaynAi.create(
    params.smbertVocab,
    params.smbertModel,
    params.qambertVocab,
    params.qambertModel,
    params.ltrModel,
    params.wasmModule,
  );

  request.sender.sendResponse(Response.ok);
}

/// The method handler for the [Method.rerank] invocation.
Future<void> rerank(ffi.XaynAi? ai, Request request) async {
  final params = RerankParams.fromJson(request.params!);
  final result = ai!.rerank(params.mode, params.histories, params.documents);
  request.sender.sendResponse(Response.fromResult(result));
}

/// The method handler for the [Method.faults] invocation.
Future<void> faults(ffi.XaynAi? ai, Request request) async {
  final result = ai!.faults();
  request.sender.sendResponse(Response.fromResult(FaultsResponse(result)));
}

/// The method handler for the [Method.serialize] invocation.
Future<void> serialize(ffi.XaynAi? ai, Request request) async {
  final result = ai!.serialize();
  request.sender.sendResponse(Response.fromResult(Uint8ListResponse(result)));
}

/// The method handler for the [Method.analytics] invocation.
Future<void> analytics(ffi.XaynAi? ai, Request request) async {
  final result = ai!.analytics();
  request.sender.sendResponse(Response.fromResult(AnalyticsResponse(result)));
}

/// The method handler for the [Method.syncdataBytes] invocation.
Future<void> syncdataBytes(ffi.XaynAi? ai, Request request) async {
  final result = ai!.syncdataBytes();
  request.sender.sendResponse(Response.fromResult(Uint8ListResponse(result)));
}

/// The method handler for the [Method.synchronize] invocation.
Future<void> synchronize(ffi.XaynAi? ai, Request request) async {
  final params = SynchronizeParams.fromJson(request.params!);
  ai!.synchronize(params.serialized);
  request.sender.sendResponse(Response.ok);
}

/// The method handler for the [Method.free] invocation.
Future<void> free(ffi.XaynAi? ai, Request request) async {
  ai!.free();
  ai = null;
  request.sender.sendResponse(Response.ok);
}

extension SendRequest on Sender {
  void sendResponse(Response response) => send(response.toJson());
}
