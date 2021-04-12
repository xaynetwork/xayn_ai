import 'dart:ffi'
    show
        AllocatorAlloc,
        Int8,
        nullptr,
        Pointer,
        StructPointer,
        Uint32,
        Uint32Pointer;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/src/doc/document.dart' show Document, History;
import 'package:xayn_ai_ffi_dart/src/doc/utils.dart'
    show FeedbackInt, RelevanceInt;
import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CDocument, CHistory;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

/// The raw document histories.
class Histories {
  late Pointer<CHistory> _hists;
  final int _size;

  /// Creates the document histories.
  Histories(List<History> histories) : _size = histories.length {
    if (_size == 0) {
      _hists = nullptr;
      return;
    }

    _hists = malloc.call<CHistory>(_size);
    for (var i = 0; i < _size; i++) {
      _hists[i].id = histories[i].id.toNativeUtf8().cast<Int8>();
      _hists[i].relevance = histories[i].relevance.toInt();
      _hists[i].feedback = histories[i].feedback.toInt();
    }
  }

  /// Gets the pointer.
  Pointer<CHistory> get ptr => _hists;

  /// Gets the size.
  int get size => _size;

  /// Frees the memory.
  void free() {
    if (_hists != nullptr) {
      for (var i = 0; i < _size; i++) {
        malloc.free(_hists[i].id);
      }
      malloc.free(_hists);
      _hists = nullptr;
    }
  }
}

/// The raw documents.
class Documents {
  late Pointer<CDocument> _docs;
  final int _size;

  /// Creates the documents.
  Documents(List<Document> documents) : _size = documents.length {
    if (_size == 0) {
      _docs = nullptr;
      return;
    }

    _docs = malloc.call<CDocument>(_size);
    for (var i = 0; i < _size; i++) {
      _docs[i].id = documents[i].id.toNativeUtf8().cast<Int8>();
      _docs[i].snippet = documents[i].snippet.toNativeUtf8().cast<Int8>();
      _docs[i].rank = documents[i].rank;
    }
  }

  /// Gets the pointer.
  Pointer<CDocument> get ptr => _docs;

  /// Gets the size.
  int get size => _size;

  /// Frees the memory.
  void free() {
    if (_docs != nullptr) {
      for (var i = 0; i < _size; i++) {
        malloc.free(_docs[i].id);
        malloc.free(_docs[i].snippet);
      }
      malloc.free(_docs);
      _docs = nullptr;
    }
  }
}

/// The ranks of the reranked documents.
class Ranks {
  Pointer<Uint32> _ranks;
  final int _size;

  /// Creates the ranks.
  Ranks(this._ranks, this._size) {
    if (_size.isNegative) {
      throw ArgumentError('negative ranks length');
    }
  }

  /// Converts the ranks to a list, which is in the same order as the documents.
  List<int> toList() => _ranks == nullptr || _size == 0
      ? List.empty()
      : _ranks.asTypedList(_size).toList(growable: false);

  /// Frees the memory.
  void free() {
    if (_ranks != nullptr) {
      ffi.ranks_drop(_ranks, _size);
      _ranks = nullptr;
    }
  }
}
