import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;

abstract class XaynAi {
  factory XaynAi(dynamic vocab, dynamic model, [dynamic? serialized]) =>
      throw UnsupportedError('Plattform not supported');
  List<int> rerank(List<History> histories, List<Document> documents) =>
      throw UnsupportedError('Plattform not supported');
  Uint8List serialize() => throw UnsupportedError('Plattform not supported');
  List<String> faults() => throw UnsupportedError('Plattform not supported');
  Analytics? analytics() => throw UnsupportedError('Plattform not supported');
  void free() => throw UnsupportedError('Plattform not supported');
}
