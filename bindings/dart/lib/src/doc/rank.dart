import 'dart:ffi' show nullptr, Pointer, StructPointer, Uint32Pointer;

import 'package:xayn_ai_ffi_dart/src/ffi/genesis.dart' show CRanks;
import 'package:xayn_ai_ffi_dart/src/ffi/library.dart' show ffi;

/// The ranks of the reranked documents.
class Ranks {
  Pointer<CRanks> _ranks;

  /// Creates the ranks.
  Ranks(this._ranks);

  /// Converts the ranks to a list, which is in the same logical order as the documents.
  List<int> toList() =>
      _ranks == nullptr || _ranks.ref.data == nullptr || _ranks.ref.len == 0
          ? List.empty()
          : _ranks.ref.data.asTypedList(_ranks.ref.len).toList(growable: false);

  /// Frees the memory.
  void free() {
    if (_ranks != nullptr) {
      ffi.ranks_drop(_ranks);
      _ranks = nullptr;
    }
  }
}
