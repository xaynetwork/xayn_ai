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
        _docs.ref.data[i].id = document.id.toNativeUtf8().cast<Uint8>();
        _docs.ref.data[i].snippet =
            document.snippet.toNativeUtf8().cast<Uint8>();
        _docs.ref.data[i].rank = document.rank;
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
          malloc.free(_docs.ref.data[i].id);
          malloc.free(_docs.ref.data[i].snippet);
        }
        malloc.free(_docs.ref.data);
      }
      malloc.free(_docs);
      _docs = nullptr;
    }
  }
}
