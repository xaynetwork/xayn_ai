import 'dart:html' show MessageEvent, Worker;
import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/web/reranker/data_provider.dart'
    show SetupData;
import 'package:xayn_ai_ffi_dart/src/web/worker/message.dart'
    show CreateArgs, FaultsReturn, Message, Method, RerankArgs, SerializeReturn;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Oneshot;

class XaynAiProxy {
  late Worker? _worker;

  static Future<XaynAiProxy> create(SetupData data,
      [Uint8List? serialized]) async {
    final worker = Worker('../worker.dart.js');

    final args = CreateArgs(
        data.smbertVocab,
        data.smbertModel,
        data.qambertVocab,
        data.qambertModel,
        data.ltrModel,
        data.wasmModule,
        data.wasmScript);

    await call(worker, Method.create, args.toJson());
    return XaynAiProxy._(worker);
  }

  XaynAiProxy._(Worker worker) {
    _worker = worker;
  }

  Future<RerankingOutcomes> rerank(RerankMode mode, List<History> histories,
      List<Document> documents) async {
    final args = RerankArgs(mode, histories, documents);
    final msg = await call(_worker!, Method.rerank, args.toJson());
    return RerankingOutcomes.fromJson(fromMessageEvent(msg));
  }

  Future<List<String>> faults() async {
    final msg = await call(_worker!, Method.faults, <String, dynamic>{});
    return FaultsReturn.fromJson(fromMessageEvent(msg)).faults;
  }

  Future<Uint8List> serialize() async {
    final msg = await call(_worker!, Method.serialize, <String, dynamic>{});
    return SerializeReturn.fromJson(fromMessageEvent(msg)).data;
  }

  Future<void> free() async {
    await call(_worker!, Method.free, <String, dynamic>{});
    _worker!.terminate();
  }
}

Future<MessageEvent> call(
    Worker worker, Method method, Map<String, dynamic> args) async {
  final channel = Oneshot();
  final message = Message(method, args, channel.sender);
  worker.postMessage(message.toJson(), [channel.sender.port]);
  return await channel.receiver.recv();
}

Map<dynamic, dynamic> fromMessageEvent(MessageEvent msg) {
  return msg.data['return'] as Map<dynamic, dynamic>;
}
