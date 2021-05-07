@JS('xayn_ai_ffi_wasm')
library web;

import 'dart:typed_data' show Uint32List, Uint8List;

import 'package:js/js.dart' show anonymous, JS;

import 'package:xayn_ai_ffi_dart/src/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/data/history.dart'
    show FeedbackCast, History, RelevanceCast;

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
  // ignore: unused_element
  external factory _Fault({int code, String message});
}

@JS()
@anonymous
class _Analytics {
  external factory _Analytics();
}

@JS('WXaynAi')
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
class XaynAi {
  late _XaynAi _ai;

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder. Optionally accepts the serialized
  /// reranker database, otherwise creates a new one.
  XaynAi(Uint8List vocab, Uint8List model, [Uint8List? serialized]) {
    _ai = _XaynAi(vocab, model, serialized);
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  ///
  /// In case of a [`Code.panic`], the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [`serialize()`].
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

    return _ai.rerank(hists, docs).toList(growable: false);
  }

  /// Serializes the current state of the reranker.
  Uint8List serialize() => _ai.serialize();

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  List<String> faults() {
    final faults = _ai.faults();

    return List.generate(
      faults.length,
      (i) => faults[i].message,
      growable: false,
    );
  }

  /// Retrieves the analytics which were collected in the penultimate reranking.
  void analytics() {
    _ai.analytics();
  }
}
