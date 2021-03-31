import 'dart:ffi' show AllocatorAlloc, Int8, nullptr, Pointer, StructPointer;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/ffi.dart'
    show CDocument, CFeedback, CHistory, CRelevance;

/// A document relevance level.
enum Relevance {
  low,
  medium,
  high,
}

extension RelevanceInt on Relevance {
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
}

/// A user feedback level.
enum Feedback {
  relevant,
  irrelevant,
  none,
}

extension FeedbackInt on Feedback {
  /// Gets the discriminant.
  int toInt() {
    switch (this) {
      case Feedback.relevant:
        return CFeedback.Relevant;
      case Feedback.irrelevant:
        return CFeedback.Irrelevant;
      case Feedback.none:
        return CFeedback.None;
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
      case CFeedback.None:
        return Feedback.none;
      default:
        throw UnsupportedError('Undefined enum variant.');
    }
  }
}

/// The raw document history.
class History {
  late Pointer<CHistory> _hist;
  final int _size;

  /// Creates the document history.
  History(
      List<String> ids, List<Relevance> relevances, List<Feedback> feedbacks)
      : _size = ids.length {
    if (_size != relevances.length || _size != feedbacks.length) {
      throw ArgumentError(
          'Document history ids, relevances and feedbacks must have the same length.');
    }

    if (_size == 0) {
      _hist = nullptr;
      return;
    }

    _hist = malloc.call<CHistory>(_size);
    for (var i = 0; i < _size; i++) {
      _hist[i].id = ids[i].toNativeUtf8().cast<Int8>();
      _hist[i].relevance = relevances[i].toInt();
      _hist[i].feedback = feedbacks[i].toInt();
    }
  }

  /// Gets the pointer.
  Pointer<CHistory> get ptr => _hist;

  /// Gets the size.
  int get size => _size;

  /// Frees the memory.
  void free() {
    if (_hist != nullptr) {
      for (var i = 0; i < _size; i++) {
        malloc.free(_hist[i].id);
      }
      malloc.free(_hist);
      _hist = nullptr;
    }
  }
}

/// The raw documents.
class Documents {
  late Pointer<CDocument> _docs;
  final int _size;

  /// Creates the documents.
  Documents(List<String> ids, List<String> snippets, List<int> ranks)
      : _size = ids.length {
    if (_size != snippets.length || _size != ranks.length) {
      throw ArgumentError(
          'Document ids, snippets and ranks must have the same length.');
    }

    if (_size == 0) {
      _docs = nullptr;
      return;
    }

    _docs = malloc.call<CDocument>(_size);
    for (var i = 0; i < _size; i++) {
      _docs[i].id = ids[i].toNativeUtf8().cast<Int8>();
      _docs[i].snippet = snippets[i].toNativeUtf8().cast<Int8>();
      _docs[i].rank = ranks[i];
    }
  }

  /// Gets the pointer.
  Pointer<CDocument> get ptr => _docs;

  /// Gets the size.
  int get size => _size;

  /// Gets the ranks.
  List<int> get ranks {
    if (_size == 0) {
      return List.empty();
    }

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
    }
  }
}
