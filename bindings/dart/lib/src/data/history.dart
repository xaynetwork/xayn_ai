import 'dart:ffi' show AllocatorAlloc, nullptr, Pointer, StructPointer, Uint8;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/src/ffi/c/genesis.dart'
    show CFeedback, CHistories, CHistory, CRelevance;

/// A document relevance level.
enum Relevance {
  low,
  medium,
  high,
}

extension RelevanceCast on Relevance {
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

  /// Creates the relevance level from a discriminant.
  static Relevance fromInt(int idx) {
    switch (idx) {
      case CRelevance.Low:
        return Relevance.low;
      case CRelevance.Medium:
        return Relevance.medium;
      case CRelevance.High:
        return Relevance.high;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }

  /// Gets the variant name.
  String toStr() {
    switch (this) {
      case Relevance.low:
        return 'Low';
      case Relevance.medium:
        return 'Medium';
      case Relevance.high:
        return 'High';
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }

  /// Creates the relevance level from a variant name.
  static Relevance fromStr(String variant) {
    switch (variant) {
      case 'Low':
        return Relevance.low;
      case 'Medium':
        return Relevance.medium;
      case 'High':
        return Relevance.high;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

/// A user feedback level.
enum Feedback {
  relevant,
  irrelevant,
  notGiven,
}

extension FeedbackCast on Feedback {
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

  /// Creates the feedback level from a discriminant.
  static Feedback fromInt(int idx) {
    switch (idx) {
      case CFeedback.Relevant:
        return Feedback.relevant;
      case CFeedback.Irrelevant:
        return Feedback.irrelevant;
      case CFeedback.NotGiven:
        return Feedback.notGiven;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }

  /// Gets the variant name.
  String toStr() {
    switch (this) {
      case Feedback.relevant:
        return 'Relevant';
      case Feedback.irrelevant:
        return 'Irrelevant';
      case Feedback.notGiven:
        return 'NotGiven';
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }

  /// Creates the feedback level from a variant name.
  static Feedback fromStr(String variant) {
    switch (variant) {
      case 'Relevant':
        return Feedback.relevant;
      case 'Irrelevant':
        return Feedback.irrelevant;
      case 'NotGiven':
        return Feedback.notGiven;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

/// The document history.
class History {
  final String _id;
  final Relevance _relevance;
  final Feedback _feedback;

  /// Creates the document history.
  History(this._id, this._relevance, this._feedback) {
    if (_id.isEmpty) {
      throw ArgumentError('empty document history id');
    }
  }

  /// Gets the id.
  String get id => _id;

  /// Gets the relevance level.
  Relevance get relevance => _relevance;

  /// Gets the user feedback.
  Feedback get feedback => _feedback;
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
        _hists.ref.data[i].id = history._id.toNativeUtf8().cast<Uint8>();
        _hists.ref.data[i].relevance = history._relevance.toInt();
        _hists.ref.data[i].feedback = history._feedback.toInt();
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
        }
        malloc.free(_hists.ref.data);
      }
      malloc.free(_hists);
      _hists = nullptr;
    }
  }
}
