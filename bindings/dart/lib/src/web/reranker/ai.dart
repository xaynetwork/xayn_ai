@JS()
library ai;

import 'dart:async' show TimeoutException;
import 'dart:html' show MessageEvent, Worker;
import 'dart:typed_data' show Uint8List;

import 'package:js/js.dart' show JS;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' as common
    show XaynAi;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/reranker/mode.dart' show RerankMode;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/common/utils.dart' show ToJson;
import 'package:xayn_ai_ffi_dart/src/web/reranker/data_provider.dart'
    show SetupData;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/request.dart'
    show CreateParams, Method, Request, RerankParams, SynchronizeParams;
import 'package:xayn_ai_ffi_dart/src/web/worker/message/response.dart'
    show AnalyticsResponse, FaultsResponse, Response, Uint8ListResponse;
import 'package:xayn_ai_ffi_dart/src/web/worker/oneshot.dart' show Oneshot;

const int kReceiveTimeoutSeconds = 15;

/// The Xayn AI.
///
/// Web worker exception handling
///
/// Exceptions other than [XaynAiException] that are thrown on the web worker
/// side are not sent back, but caught and their error message is logged in
/// the console. In this case, the called method/static function of [XaynAi]
/// throws a [TimeoutException]. The timeout is set to [kReceiveTimeoutSeconds].
///
/// After a [TimeoutException] was thrown, the instance must be disposed by
/// calling [XaynAi.free]. The instance must not be used afterwards.
/// Note: Calling [XaynAi.free] can also throw a [TimeoutException].
class XaynAi implements common.XaynAi {
  final Worker? _worker;

  /// Creates and initializes the Xayn AI from a given state.
  ///
  /// Requires the necessary [SetupData] and the state.
  /// It will throw an error if the provided state is empty.
  static Future<XaynAi> restore(SetupData data, Uint8List serialized) async {
    if (serialized.isEmpty) {
      throw ArgumentError('Serialized state cannot be empty');
    }
    return await XaynAi._create(data, serialized);
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the necessary [SetupData] for the AI.
  static Future<XaynAi> create(SetupData data) async {
    return await XaynAi._create(data, null);
  }

  static Future<XaynAi> _create(SetupData data, Uint8List? serialized) async {
    final worker = Worker(data.webWorkerScript);

    final params = CreateParams(
      data.smbertVocab,
      data.smbertModel,
      data.qambertVocab,
      data.qambertModel,
      data.ltrModel,
      data.wasmModule,
      data.wasmScript,
      serialized,
    );

    try {
      final response = await _call(worker, Method.create, params: params);
      if (response.isException()) {
        throw response.result!;
      }
    } catch (_) {
      worker.terminate();
      rethrow;
    }

    return XaynAi._(worker);
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder and the LTR model.
  /// Optionally accepts the serialized reranker database, otherwise creates a new one.
  XaynAi._(this._worker);

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<RerankingOutcomes> rerank(
    RerankMode mode,
    List<History> histories,
    List<Document> documents,
  ) async {
    final response = await _call(
      _worker!,
      Method.rerank,
      params: RerankParams(mode, histories, documents),
    );

    if (response.isException()) {
      throw response.result!;
    }

    return RerankingOutcomes.fromJson(response.result!);
  }

  /// Serializes the current state of the reranker.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<Uint8List> serialize() async {
    final response = await _call(_worker!, Method.serialize);

    if (response.isException()) {
      throw response.result!;
    }

    return Uint8ListResponse.fromJson(response.result!).data;
  }

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<List<String>> faults() async {
    final response = await _call(_worker!, Method.faults);

    if (response.isException()) {
      throw response.result!;
    }

    return FaultsResponse.fromJson(response.result!).faults;
  }

  /// Retrieves the analytics which were collected in the penultimate reranking.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<Analytics?> analytics() async {
    final response = await _call(_worker!, Method.analytics);

    if (response.isException()) {
      throw response.result!;
    }

    return AnalyticsResponse.fromJson(response.result!).analytics;
  }

  /// Serializes the synchronizable data of the reranker.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<Uint8List> syncdataBytes() async {
    final response = await _call(_worker!, Method.syncdataBytes);

    if (response.isException()) {
      throw response.result!;
    }

    return Uint8ListResponse.fromJson(response.result!).data;
  }

  /// Synchronizes the internal data of the reranker with another.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<void> synchronize(Uint8List serialized) async {
    final response = await _call(
      _worker!,
      Method.synchronize,
      params: SynchronizeParams(serialized),
    );

    if (response.isException()) {
      throw response.result!;
    }
  }

  /// Frees the memory.
  @override
  Future<void> free() async {
    try {
      final response = await _call(_worker!, Method.free);
      if (response.isException()) {
        throw response.result!;
      }
    } catch (_) {
      rethrow;
    } finally {
      _worker!.terminate();
    }
  }
}

Future<Response> _call<P extends ToJson>(Worker worker, Method method,
    {P? params}) async {
  final channel = Oneshot();
  final sender = channel.sender;
  final request = Request(method, params?.toJson(), sender);
  worker.postMessage(request.toJson(), [sender.port!]);
  final receiver = channel.receiver;

  MessageEvent? msg;
  try {
    msg = await receiver
        .recv()
        .timeout(Duration(seconds: kReceiveTimeoutSeconds));
  } on TimeoutException {
    receiver.close();
    rethrow;
  }

  return Response.fromJson(msg.data as Map<dynamic, dynamic>);
}
