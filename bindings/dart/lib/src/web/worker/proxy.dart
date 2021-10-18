import 'dart:html' show Worker;
import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' as common
    show XaynAi;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart';
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/web/reranker/data_provider.dart'
    show SetupData;
import 'package:xayn_ai_ffi_dart/src/web/worker/message.dart'
    show
        CreateParams,
        FaultsResponse,
        Method,
        Request,
        RerankParams,
        Response,
        SerializeResponse;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Oneshot;

class XaynAiWorker implements common.XaynAi {
  late Worker? _worker;

  static Future<XaynAiWorker> create(SetupData data,
      [Uint8List? serialized]) async {
    final worker = Worker('../worker.dart.js');

    final params = CreateParams(
        data.smbertVocab,
        data.smbertModel,
        data.qambertVocab,
        data.qambertModel,
        data.ltrModel,
        data.wasmModule,
        data.wasmScript,
        serialized);

    final response = await call(worker, Method.create, params: params.toJson());
    if (response.isError()) {
      throw response.error!;
    } else {
      return XaynAiWorker._(worker);
    }
  }

  XaynAiWorker._(Worker worker) {
    _worker = worker;
  }

  @override
  Future<RerankingOutcomes> rerank(RerankMode mode, List<History> histories,
      List<Document> documents) async {
    final response = await call(_worker!, Method.rerank,
        params: RerankParams(mode, histories, documents).toJson());
    if (response.isError()) {
      throw response.error!;
    } else {
      return RerankingOutcomes.fromJson(response.result!);
    }
  }

  @override
  Future<List<String>> faults() async {
    final response = await call(_worker!, Method.faults);
    if (response.isError()) {
      throw response.error!;
    } else {
      return FaultsResponse.fromJson(response.result!).faults;
    }
  }

  @override
  Future<Uint8List> serialize() async {
    final response = await call(_worker!, Method.serialize);
    if (response.isError()) {
      throw response.error!;
    } else {
      return SerializeResponse.fromJson(response.result!).data;
    }
  }

  @override
  Future<Analytics?> analytics() {
    throw UnimplementedError();
  }

  @override
  Future<Uint8List> syncdataBytes() {
    throw UnimplementedError();
  }

  @override
  Future<void> synchronize(Uint8List serialized) {
    throw UnimplementedError();
  }

  @override
  Future<void> free() async {
    final response = await call(_worker!, Method.free);
    _worker!.terminate();

    if (response.isError()) {
      throw response.error!;
    }
  }
}

Future<Response> call(Worker worker, Method method,
    {Map<String, dynamic>? params}) async {
  final channel = Oneshot();
  final sender = channel.sender;
  final request = Request(method, params, sender);
  worker.postMessage(request.toJson(), [sender.port!]);
  final msg = await channel.receiver.recv();
  return Response.fromJson(msg.data as Map<dynamic, dynamic>);
}
