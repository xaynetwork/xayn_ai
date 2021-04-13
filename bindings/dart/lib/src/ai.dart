import 'dart:ffi'
    show
        Int8,
        nullptr,
        Pointer,
        Uint8,
        AllocatorAlloc,
        // ignore: unused_shown_name
        Uint8Pointer;

import 'dart:typed_data' show Uint8List;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/src/doc/document.dart' show Document, History;
import 'package:xayn_ai_ffi_dart/src/doc/documents.dart'
    show Documents, Histories, Ranks, Bytes;
import 'package:xayn_ai_ffi_dart/src/error.dart' show XaynAiError;
import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CXaynAi;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

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
  XaynAi(Uint8List serialized, String vocab, String model) {
    final vocabPtr = vocab.toNativeUtf8().cast<Int8>();
    final modelPtr = model.toNativeUtf8().cast<Int8>();
    final error = XaynAiError();

    Pointer<Uint8> serializedPtr = nullptr;
    var serializedLen = 0;

    try {
      if (serialized.isNotEmpty) {
        serializedPtr = malloc.call<Uint8>(serializedLen);
        serialized.asMap().forEach((i, byte) {
          serializedPtr[i] = byte;
        });

        serializedLen = serialized.length;
      }

      _ai = ffi.xaynai_new(
          serializedPtr, serializedLen, vocabPtr, modelPtr, error.ptr);

      if (error.isError()) {
        throw error.toException();
      }
    } finally {
      malloc.free(vocabPtr);
      malloc.free(modelPtr);
      error.free();
      if (serializedPtr != nullptr) {
        malloc.free(serializedPtr);
      }
    }
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  List<int> rerank(List<History> histories, List<Document> documents) {
    final hists = Histories(histories);
    final docs = Documents(documents);
    final error = XaynAiError();

    final ranks = Ranks(
      ffi.xaynai_rerank(
          _ai, hists.ptr, hists.size, docs.ptr, docs.size, error.ptr),
      docs.size,
    );
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

  /// Frees the memory.
  void free() {
    if (_ai != nullptr) {
      ffi.xaynai_drop(_ai);
      _ai = nullptr;
    }
  }
}
