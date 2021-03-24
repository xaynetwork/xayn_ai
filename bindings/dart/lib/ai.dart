import 'dart:ffi' show DynamicLibrary, Int8, nullptr, Pointer;
import 'dart:io' show Platform;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer, Utf8, Utf8Pointer;

import 'package:xayn_ai_ffi_dart/document.dart' show Documents;
import 'package:xayn_ai_ffi_dart/error.dart' show XaynAiError, XaynAiException;
import 'package:xayn_ai_ffi_dart/ffi.dart' show CXaynAi, XaynAiFfi;

extension XaynAiFfiImpl on XaynAiFfi {
  static DynamicLibrary load() {
    if (Platform.isAndroid) {
      return DynamicLibrary.open('libxayn_ai_ffi_c.so');
    }
    if (Platform.isIOS) {
      return DynamicLibrary.process();
    }
    if (Platform.isLinux) {
      return DynamicLibrary.open('../target/debug/libxayn_ai_ffi_c.so');
    }
    if (Platform.isMacOS) {
      return DynamicLibrary.open('../target/debug/libxayn_ai_ffi_c.dylib');
    }
    throw UnsupportedError('Unsupported platform.');
  }
}

final XaynAiFfi xaynAiFfi = XaynAiFfi(XaynAiFfiImpl.load());

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
      _ai = xaynAiFfi.xaynai_new(vocabPtr, modelPtr, error.ptr);
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
      xaynAiFfi.xaynai_rerank(_ai, docs.ptr, docs.size, error.ptr);
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
      xaynAiFfi.xaynai_drop(_ai);
      _ai = nullptr;
    }
  }
}
