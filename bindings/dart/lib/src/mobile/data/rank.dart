import 'dart:ffi' show nullptr, Pointer, StructPointer, Uint32Pointer;

import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CBoxedSlice_u32;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/library.dart' show ffi;

/// The ranks of the reranked documents.
class Ranks {
  late Pointer<CBoxedSlice_u32> _ranks;

  /// Creates the ranks.
  ///
  /// This constructor never throws an exception.
  Ranks(this._ranks);

  /// Converts the ranks to a list, which is in the same logical order as the documents.
  List<int> toList() {
    assert(
      _ranks == nullptr || _ranks.ref.data != nullptr,
      'unexpected ranks pointer state',
    );

    return _ranks == nullptr
        ? List.empty()
        : _ranks.ref.data.asTypedList(_ranks.ref.len).toList(growable: false);
  }

  /// Frees the memory.
  void free() {
    assert(
      _ranks == nullptr || _ranks.ref.data != nullptr,
      'unexpected ranks pointer state',
    );

    if (_ranks != nullptr) {
      ffi.ranks_drop(_ranks);
      _ranks = nullptr;
    }
  }
}
