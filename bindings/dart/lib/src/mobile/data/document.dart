import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer, Uint8;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CDocument, CDocuments;

/// The raw documents.
class Documents {
  late Pointer<CDocuments> _docs;

  /// Creates the documents.
  ///
  /// This constructor never throws an exception.
  Documents(List<Document> documents) {
    _docs = malloc.call<CDocuments>();
    _docs.ref.len = documents.length;
    if (documents.isEmpty) {
      _docs.ref.data = nullptr;
    } else {
      _docs.ref.data = malloc.call<CDocument>(_docs.ref.len);
      documents.asMap().forEach((i, document) {
        var cdoc = _docs.ref.data[i];
        cdoc.id = document.id.toNativeUtf8().cast<Uint8>();
        cdoc.snippet = document.snippet.toNativeUtf8().cast<Uint8>();
        cdoc.rank = document.rank;
        cdoc.session = document.session.toNativeUtf8().cast<Uint8>();
        cdoc.query_count = document.queryCount;
        cdoc.query_id = document.queryId.toNativeUtf8().cast<Uint8>();
        cdoc.query_words = document.queryWords.toNativeUtf8().cast<Uint8>();
        cdoc.url = document.url.toNativeUtf8().cast<Uint8>();
        cdoc.domain = document.domain.toNativeUtf8().cast<Uint8>();
      });
    }
  }

  /// Gets the pointer.
  Pointer<CDocuments> get ptr => _docs;

  /// Frees the memory.
  void free() {
    if (_docs != nullptr) {
      if (_docs.ref.data != nullptr) {
        for (var i = 0; i < _docs.ref.len; i++) {
          var cdoc = _docs.ref.data[i];
          malloc.free(cdoc.id);
          malloc.free(cdoc.snippet);
          malloc.free(cdoc.session);
          malloc.free(cdoc.query_id);
          malloc.free(cdoc.query_words);
          malloc.free(cdoc.url);
          malloc.free(cdoc.domain);
        }
        malloc.free(_docs.ref.data);
      }
      malloc.free(_docs);
      _docs = nullptr;
    }
  }
}
