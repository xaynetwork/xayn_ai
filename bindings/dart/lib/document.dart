import 'dart:ffi' show AllocatorAlloc, Int8, nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/ffi.dart' show CDocument;

class Documents {
  late Pointer<CDocument> _docs;
  late int _size;

  /// Gets the pointer.
  Pointer<CDocument> get ptr => _docs;

  /// Gets the size.
  int get size => _size;

  /// Creates the documents.
  Documents(List<String> ids, List<String> snippets, List<int> ranks) {
    _size = ids.length;
    if (_size < 1 || _size != snippets.length || _size != ranks.length) {
      throw ArgumentError(
          'Document ids, snippets and ranks must have the same positive length.');
    }

    _docs = malloc.call<CDocument>(_size);
    for (var i = 0; i < _size; i++) {
      _docs[i].id = ids[i].toNativeUtf8().cast<Int8>();
      _docs[i].snippet = snippets[i].toNativeUtf8().cast<Int8>();
      _docs[i].rank = ranks[i];
    }
  }

  /// Gets the ranks.
  List<int> get ranks {
    if (_docs == nullptr) {
      throw ArgumentError('Documents were already freed.');
    }

    return List.generate(_size, (i) => _docs[i].rank, growable: false);
  }

  /// Frees the memory.
  void free() {
    if (_docs != nullptr) {
      for (var i = 0; i < _size; i++) {
        malloc.free(_docs[i].id);
        malloc.free(_docs[i].snippet);
      }
      malloc.free(_docs);
      _docs = nullptr;
      _size = 0;
    }
  }
}
