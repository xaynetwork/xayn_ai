import 'dart:ffi' show Int8, nullptr, Pointer;
import 'dart:typed_data' show Uint8List;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/src/data/document.dart'
    show Document, Documents;
import 'package:xayn_ai_ffi_dart/src/data/history.dart' show Histories, History;
import 'package:xayn_ai_ffi_dart/src/data/rank.dart' show Ranks;
import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CXaynAi;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;
import 'package:xayn_ai_ffi_dart/src/reranker/analytics.dart' show Analytics;
import 'package:xayn_ai_ffi_dart/src/reranker/bytes.dart' show Bytes;
import 'package:xayn_ai_ffi_dart/src/result/error.dart' show XaynAiError;
import 'package:xayn_ai_ffi_dart/src/result/warning.dart' show Warnings;

/// The Xayn AI.
///
/// # Examples
/// - Create a Xayn AI with [`XaynAi()`].
/// - Rerank documents with [`rerank()`].
/// - Free memory with [`free()`].
class XaynAi {
  late Pointer<CXaynAi> _ai;

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the vocabulary and model of the tokenizer/embedder. Optionally accepts the serialized
  /// reranker database, otherwise creates a new one.
  XaynAi(String vocab, String model, [Uint8List? serialized]) {
    final vocabPtr = vocab.toNativeUtf8().cast<Int8>();
    final modelPtr = model.toNativeUtf8().cast<Int8>();
    final bytes = Bytes.fromList(serialized ?? Uint8List(0));
    final error = XaynAiError();

    _ai = ffi.xaynai_new(vocabPtr, modelPtr, bytes.ptr, error.ptr);
    try {
      if (error.isError()) {
        throw error.toException();
      }
    } finally {
      malloc.free(vocabPtr);
      malloc.free(modelPtr);
      bytes.free();
      error.free();
    }
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  List<int> rerank(List<History> histories, List<Document> documents) {
    final hists = Histories(histories);
    final docs = Documents(documents);
    final error = XaynAiError();

    final ranks = Ranks(ffi.xaynai_rerank(_ai, hists.ptr, docs.ptr, error.ptr));
    try {
      if (error.isError()) {
        throw error.toException();
      }
      return ranks.toList();
    } finally {
      hists.free();
      docs.free();
      error.free();
      ranks.free();
    }
  }

  /// Serializes the current state of the reranker.
  Uint8List serialize() {
    final error = XaynAiError();

    final bytes = Bytes(ffi.xaynai_serialize(_ai, error.ptr));
    try {
      if (error.isError()) {
        throw error.toException();
      }
      return bytes.toList();
    } finally {
      error.free();
      bytes.free();
    }
  }

  /// Retrieves warnings which might occur during reranking.
  List<String> warnings() {
    final error = XaynAiError();

    final warnings = Warnings(ffi.xaynai_warnings(_ai, error.ptr));
    try {
      if (error.isError()) {
        throw error.toException();
      }
      return warnings.toList();
    } finally {
      error.free();
      warnings.free();
    }
  }

  /// Retrieves the analytics which were collected in the penultimate reranking.
  void analytics() {
    final error = XaynAiError();

    final analytics = Analytics(ffi.xaynai_analytics(_ai, error.ptr));
    try {
      if (error.isError()) {
        throw error.toException();
      }
      return;
    } finally {
      error.free();
      analytics.free();
    }
  }

  /// Frees the memory.
  void free() {
    if (_ai != nullptr) {
      ffi.xaynai_drop(_ai);
      _ai = nullptr;
    }
  }
}
