import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer, Uint8;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/src/common/data/history.dart'
    show Feedback, History, Relevance, DayOfWeek, UserAction;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart'
    show CFeedback, CHistories, CHistory, CRelevance, CDayOfWeek, CUserAction;

extension RelevanceToInt on Relevance {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Relevance.low:
        return CRelevance.Low;
      case Relevance.medium:
        return CRelevance.Medium;
      case Relevance.high:
        return CRelevance.High;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

extension FeedbackToInt on Feedback {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Feedback.relevant:
        return CFeedback.Relevant;
      case Feedback.irrelevant:
        return CFeedback.Irrelevant;
      case Feedback.notGiven:
        return CFeedback.NotGiven;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

extension DayOfWeekToInt on DayOfWeek {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case DayOfWeek.mon:
        return CDayOfWeek.Mon;
      case DayOfWeek.tue:
        return CDayOfWeek.Tue;
      case DayOfWeek.wed:
        return CDayOfWeek.Wed;
      case DayOfWeek.thu:
        return CDayOfWeek.Thu;
      case DayOfWeek.fri:
        return CDayOfWeek.Fri;
      case DayOfWeek.sat:
        return CDayOfWeek.Sat;
      case DayOfWeek.sun:
        return CDayOfWeek.Sun;
    }
  }
}

extension UserActionToInt on UserAction {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case UserAction.miss:
        return CUserAction.Miss;
      case UserAction.skip:
        return CUserAction.Skip;
      case UserAction.click:
        return CUserAction.Click;
    }
  }
}

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
        _hists.ref.data[i].id = history.id.toNativeUtf8().cast<Uint8>();
        _hists.ref.data[i].relevance = history.relevance.toInt();
        _hists.ref.data[i].feedback = history.feedback.toInt();
        _hists.ref.data[i].session =
            history.session.toNativeUtf8().cast<Uint8>();
        _hists.ref.data[i].query_count = history.queryCount;
        _hists.ref.data[i].query_id =
            history.queryId.toNativeUtf8().cast<Uint8>();
        _hists.ref.data[i].query_words =
            history.queryWords.toNativeUtf8().cast<Uint8>();
        _hists.ref.data[i].day = history.day.toInt();
        _hists.ref.data[i].url = history.url.toNativeUtf8().cast<Uint8>();
        _hists.ref.data[i].domain = history.domain.toNativeUtf8().cast<Uint8>();
        _hists.ref.data[i].rank = history.rank;
        _hists.ref.data[i].user_action = history.userAction.toInt();
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
          malloc.free(_hists.ref.data[i].id);
          malloc.free(_hists.ref.data[i].session);
          malloc.free(_hists.ref.data[i].query_id);
          malloc.free(_hists.ref.data[i].query_words);
          malloc.free(_hists.ref.data[i].url);
          malloc.free(_hists.ref.data[i].domain);
        }
        malloc.free(_hists.ref.data);
      }
      malloc.free(_hists);
      _hists = nullptr;
    }
  }
}
