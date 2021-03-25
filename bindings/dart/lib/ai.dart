import 'dart:ffi' show DynamicLibrary, Int8, nullptr, Pointer;
import 'dart:io' show Platform;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/document.dart' show Documents;
import 'package:xayn_ai_ffi_dart/error.dart' show XaynAiError, XaynAiException;
import 'package:xayn_ai_ffi_dart/ffi.dart' show CXaynAi, XaynAiFfi;

final XaynAiFfi ffi = XaynAiFfi(Platform.isAndroid
    ? DynamicLibrary.open('libxayn_ai_ffi_c.so')
    : Platform.isIOS
        ? DynamicLibrary.process()
        : Platform.isLinux
            ? DynamicLibrary.open('../target/debug/libxayn_ai_ffi_c.so')
            : Platform.isMacOS
                ? DynamicLibrary.open('../target/debug/libxayn_ai_ffi_c.dylib')
                : throw UnsupportedError('Unsupported platform.'));

/// The Xayn AI.
///
/// # Examples
/// - Create a Xayn AI with [`XaynAi()`].
/// - Rerank documents with [`rerank()`].
/// - Free memory with [`free()`].
class XaynAi {
  late Pointer<CXaynAi> _ai;

  /// Gets the pointer.
  Pointer<CXaynAi> get ptr => _ai;

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
        throw XaynAiException(error);
      }
    } finally {
      malloc.free(vocabPtr);
      malloc.free(modelPtr);
      error.free();
    }
  }

  /// Reranks the documents.
  List<int> rerank(List<String> ids, List<String> snippets, List<int> ranks) {
    final docs = Documents(ids, snippets, ranks);
    final error = XaynAiError();

    try {
      ffi.xaynai_rerank(_ai, docs.ptr, docs.size, error.ptr);
      if (error.isSuccess()) {
        return docs.ranks;
      } else {
        throw XaynAiException(error);
      }
    } finally {
      docs.free();
      error.free();
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
