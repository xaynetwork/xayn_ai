import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer, Uint8;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show
        FeedbackToInt,
        History,
        RelevanceToInt,
        DayOfWeekToInt,
        UserActionToInt;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CHistories, CHistory;

/// The raw document histories.
class Histories {
  late Pointer<CHistories> _hists;

  /// Creates the document histories.
  ///
  /// This constructor never throws an exception.
  Histories(List<History> histories) {
    _hists = malloc.call<CHistories>();
    _hists.ref.len = histories.length;
    if (histories.isEmpty) {
      _hists.ref.data = nullptr;
    } else {
      _hists.ref.data = malloc.call<CHistory>(_hists.ref.len);
      histories.asMap().forEach((i, history) {
        var chist = _hists.ref.data[i];
        chist.id = history.id.toNativeUtf8().cast<Uint8>();
        chist.relevance = history.relevance.toInt();
        chist.feedback = history.feedback.toInt();
        chist.session = history.session.toNativeUtf8().cast<Uint8>();
        chist.query_count = history.queryCount;
        chist.query_id = history.queryId.toNativeUtf8().cast<Uint8>();
        chist.query_words = history.queryWords.toNativeUtf8().cast<Uint8>();
        chist.day = history.day.toInt();
        chist.url = history.url.toNativeUtf8().cast<Uint8>();
        chist.domain = history.domain.toNativeUtf8().cast<Uint8>();
        chist.rank = history.rank;
        chist.user_action = history.userAction.toInt();
      });
    }
  }

  /// Gets the pointer.
  Pointer<CHistories> get ptr => _hists;

  /// Frees the memory.
  void free() {
    if (_hists != nullptr) {
      if (_hists.ref.data != nullptr) {
        for (var i = 0; i < _hists.ref.len; i++) {
          var chist = _hists.ref.data[i];
          malloc.free(chist.id);
          malloc.free(chist.session);
          malloc.free(chist.query_id);
          malloc.free(chist.query_words);
          malloc.free(chist.url);
          malloc.free(chist.domain);
        }
        malloc.free(_hists.ref.data);
      }
      malloc.free(_hists);
      _hists = nullptr;
    }
  }
}
