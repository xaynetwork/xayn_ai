@JS()
library web;

import 'dart:typed_data' show Uint32List, Uint8List;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/data/history.dart'
    show FeedbackStr, History, RelevanceStr;
import 'package:xayn_ai_ffi_dart/src/reranker/analytics.dart' show Analytics;
import 'package:xayn_ai_ffi_dart/src/reranker/base.dart' as base;

@JS('WebAssembly.RuntimeError')
class _RuntimeException {}

@JS()
@anonymous
class _History {
  external factory _History({
    String id,
    String relevance,
    // ignore: non_constant_identifier_names
    String user_feedback,
  });
}

@JS()
@anonymous
class _Document {
  external factory _Document({String id, int rank, String snippet});
}

@JS()
@anonymous
class _Fault {
  external String get message;
  external factory _Fault({
    // ignore: unused_element
    int code,
    // ignore: unused_element
    String message,
  });
}

@JS()
@anonymous
class _Analytics {
  // ignore: non_constant_identifier_names
  external double get ndcg_ltr;
  // ignore: non_constant_identifier_names
  external double get ndcg_context;
  // ignore: non_constant_identifier_names
  external double get ndcg_initial_ranking;
  // ignore: non_constant_identifier_names
  external double get ndcg_final_ranking;
  external factory _Analytics({
    // ignore: non_constant_identifier_names, unused_element
    double ndcg_ltr,
    // ignore: non_constant_identifier_names, unused_element
    double ndcg_context,
    // ignore: non_constant_identifier_names, unused_element
    double ndcg_initial_ranking,
    // ignore: non_constant_identifier_names, unused_element
    double ndcg_final_ranking,
  });
}

@JS('xayn_ai_ffi_wasm.WXaynAi')
class _XaynAi {
  external _XaynAi(Uint8List vocab, Uint8List model, [Uint8List? serialized]);
  external Uint32List rerank(
    List<_History> history,
    List<_Document> documents,
  );
  external Uint8List serialize();
  external List<_Fault> faults();
  external _Analytics analytics();
}

/// The Xayn AI.
class XaynAi implements base.XaynAi {
  late _XaynAi _ai;

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder. Optionally accepts the serialized
  /// reranker database, otherwise creates a new one.
  XaynAi(Uint8List vocab, Uint8List model, [Uint8List? serialized]) {
    try {
      _ai = _XaynAi(vocab, model, serialized);
    } on _RuntimeException {
      throw Exception('WebAssembly RuntimeError');
    } catch (exception) {
      rethrow;
    }
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
  @override
  List<int> rerank(List<History> histories, List<Document> documents) {
    final hists = List.generate(
      histories.length,
      (i) => _History(
        id: histories[i].id,
        relevance: histories[i].relevance.toStr(),
        user_feedback: histories[i].feedback.toStr(),
      ),
      growable: false,
    );
    final docs = List.generate(
      documents.length,
      (i) => _Document(
        id: documents[i].id,
        rank: documents[i].rank,
        snippet: documents[i].snippet,
      ),
      growable: false,
    );

    late final Uint32List ranks;
    try {
      ranks = _ai.rerank(hists, docs);
    } on _RuntimeException {
      throw Exception('WebAssembly RuntimeError');
    } catch (exception) {
      rethrow;
    }

    return ranks.toList(growable: false);
  }

  /// Serializes the current state of the reranker.
  @override
  Uint8List serialize() {
    try {
      return _ai.serialize();
    } on _RuntimeException {
      throw Exception('WebAssembly RuntimeError');
    } catch (exception) {
      rethrow;
    }
  }

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  @override
  List<String> faults() {
    late final List<_Fault> faults;
    try {
      faults = _ai.faults();
    } on _RuntimeException {
      throw Exception('WebAssembly RuntimeError');
    } catch (exception) {
      rethrow;
    }

    return List.generate(
      faults.length,
      (i) => faults[i].message,
      growable: false,
    );
  }

  /// Retrieves the analytics which were collected in the penultimate reranking.
  @override
  Analytics? analytics() {
    late final _Analytics analytics;
    try {
      analytics = _ai.analytics();
    } on _RuntimeException {
      throw Exception('WebAssembly RuntimeError');
    } catch (exception) {
      rethrow;
    }

    return Analytics(
      analytics.ndcg_ltr,
      analytics.ndcg_context,
      analytics.ndcg_initial_ranking,
      analytics.ndcg_final_ranking,
    );
  }

  /// Frees the memory.
  @override
  void free() {}
}
