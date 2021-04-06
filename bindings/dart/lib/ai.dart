import 'dart:ffi' show Int8, nullptr, Pointer;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/document.dart'
    show Documents, Feedback, History, Ranks, Relevance;
import 'package:xayn_ai_ffi_dart/error.dart' show XaynAiError;
import 'package:xayn_ai_ffi_dart/ffi.dart' show CXaynAi;
import 'package:xayn_ai_ffi_dart/library.dart' show ffi;

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
  /// Requires the paths to the vocabulary and model files.
  XaynAi(String vocab, String model) {
    final vocabPtr = vocab.toNativeUtf8().cast<Int8>();
    final modelPtr = model.toNativeUtf8().cast<Int8>();
    final error = XaynAiError();

    try {
      _ai = ffi.xaynai_new(vocabPtr, modelPtr, error.ptr);
      if (!error.isSuccess()) {
        throw error.toException();
      }
    } finally {
      malloc.free(vocabPtr);
      malloc.free(modelPtr);
      error.free();
    }
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  List<int> rerank(
      List<String> histIds,
      List<Relevance> histRelevances,
      List<Feedback> histFeedbacks,
      List<String> docsIds,
      List<String> docsSnippets,
      List<int> docsRanks) {
    final hist = History(histIds, histRelevances, histFeedbacks);
    final docs = Documents(docsIds, docsSnippets, docsRanks);
    final error = XaynAiError();
    Ranks? ranks;

    try {
      ranks = Ranks(
        ffi.xaynai_rerank(
            _ai, hist.ptr, hist.size, docs.ptr, docs.size, error.ptr),
        docs.size,
      );
      if (error.isSuccess()) {
        return ranks.toList();
      } else {
        throw error.toException();
      }
    } finally {
      hist.free();
      docs.free();
      error.free();
      ranks?.free();
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
